#!/usr/bin/env python3
# coding: utf-8

import argparse
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from preprocess.preperator import (
    DatasetPreparator,
    process_datasets,
    whisper_max_target_positions,
)

# Split on : but allow : inside [] for the HF split slicing syntax
# https://huggingface.co/docs/datasets/loading#slice-splits
dataset_spec_split_pattern = r":(?=(?:[^\[\]]|\[[^\[\]]*\])*$)"


def load_datasets(dataset_specs):
    datasets = []
    for spec in dataset_specs:
        parts = re.split(dataset_spec_split_pattern, spec)

        dataset_name = parts[0]
        split = parts[1] if len(parts) == 2 else "train"

        dataset = load_dataset(dataset_name, split=split)
        datasets.append(dataset)
    return datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Ensure input_features are decompressed if needed:
        input_features = []
        for feature in features:
            pad_amount = feature.get("pad_amount", 0)
            if pad_amount > 0:
                pad_value = feature["pad_value"]  # (d)
                pad_tensor = torch.tensor([pad_value] * pad_amount).T  # (d, pad_amount)
                base_features = torch.tensor(feature["input_features"])  # (d, feat_len)
                final_features = torch.concatenate([base_features, pad_tensor], dim=-1)  # (d, feat_len + pad_amount)
                input_features.append(final_features)
            else:
                input_features.append(torch.tensor(feature["input_features"]))

        batch = BatchFeature({"input_features": torch.stack(input_features)})

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"]

        # Labels, represent the input to the decoder
        batch["decoder_input_ids"] = labels[:, :-1]

        # Shift all labels to the left, thus the expected generated label
        # is at the same index of the generated output id from the decoder
        # and the loss function would compare them (cross entropy loss in this case)
        # Note - this means there is no loss calculated for the first "start of transcript" token id
        # since it is not expected to be predicted but always provided.
        # The loss is calculated for the task/lang/notimestamp tokens since the model needs to know
        # to associate them with the proper output
        # **Warning!** the labels are shifted here, and some version of transformers will assume
        # they are not if using the default "ForCausalLMLoss"
        # Once Whisper is updated to use that built-in loss - need to reconsider the collator.
        # Atm the custom loss function expects this shift to be done here.
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # Where we do not need to attend when calculating loss - -100 is the agreed
        # ignored value for the pytorch loss functions
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor, metric, normalizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace the loss-ignored value with the padding token for this model
    # which would be decoded to an empty string
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]

    wer = metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


def prepare_model_for_qlora(model):
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=64,
        lora_alpha=1,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"],
        # modules_to_save=["embed_tokens"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


def compute_loss_func(
    outputs: Seq2SeqLMOutput,
    labels: torch.Tensor,
    num_items_in_batch: int,
):
    # Until the Whisper model loss is updated to use the new Transfomers loss infrastruture,
    # it suffers from  bug in how grad acc steps loss is calculated. This is a workaround.
    # See https://huggingface.co/blog/gradient_accumulation

    lm_logits = outputs.logits
    vocab_size = lm_logits.shape[2]
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    # move labels to correct device to enable PP
    labels = labels.to(lm_logits.device)

    loss = loss_fct(lm_logits.view(-1, vocab_size), labels.reshape(-1))
    if reduction == "sum":
        loss = loss / num_items_in_batch

    return loss


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Whisper model with custom datasets.")
    parser.add_argument(
        "--train_datasets",
        nargs="*",
        help="Dataset(s) to train on. Format: dataset_name[:split_name]",
    )
    parser.add_argument("--save_processed", help="Dataset name to save processed data (will save both train and eval)")
    parser.add_argument(
        "--include_timestamps_prob",
        type=float,
        default=0.5,
        help="Probability to include timestamps with a sample (This might be a synthetic augmentation or an existing transcription timestamps)",
    )
    parser.add_argument(
        "--include_prev_text_prob",
        type=float,
        default=0.5,
        help="Probability to include previous text with a sample only when prev transcript is present on the sample",
    )
    parser.add_argument(
        "--inject_synthetic_timestamps",
        help="If timestamps are to be included with a sample but not provided, a start+end timestamp token will be injected",
        action="store_true",
    )
    parser.add_argument(
        "--audio_shift_augmentation",
        help="When timestamps are injected, also randomize shift augmentation on it",
        action="store_true",
    )
    parser.add_argument(
        "--use_preprocessed",
        nargs="+",
        help="Dataset name to load preprocessed data from (either local path or remote dataset)",
    )
    parser.add_argument(
        "--use_preprocessed_probs", nargs="+", type=float, help="Probability of using preprocessed data"
    )
    parser.add_argument(
        "--ds_processor_proc_num", type=int, default=1, help="Number of parallel processors for datasets preparation"
    )
    parser.add_argument("--model_name", default="openai/whisper-large-v2", help="Name of the model to train")
    parser.add_argument("--output_model_name", required=True, help="Name of the fine-tuned model to generate")
    parser.add_argument("--hf_org_name", default="ivrit-ai", help="Name of HF Org to push the model to")
    parser.add_argument("--skip_push_to_hub", action="store_true", help="Don't push result model to hub")
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        help="Reference dataset(s) for evaluation. Format: dataset_name[:split_name]",
    )
    parser.add_argument(
        "--save_only_model", action="store_true", default=False, help="Save only the model without optimizer state"
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep during training",
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", help="Try and resuming for last saved checkpoint"
    )
    parser.add_argument(
        "--resume_from_checkpoint_path", type=str, help="Path to checkpoint to resume from", default=None
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between each model save/upload.")
    parser.add_argument(
        "--ignore_data_skip", action="store_true", help="Ignore data skip when resuming from checkpoint"
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["bf16", "fp16", "tf32", None],
        default=None,
        help="Mixed precision mode for training",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["sdpa"],
        help="Attention implementation to use (only 'sdpa' available)",
    )
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="How many steps to train for - overrides num_train_epochs"
    )
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="constant_with_warmup", help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--eval_steps", type=int, help="Number of steps between two evals, if not specified defaults to logging_steps."
    )
    parser.add_argument(
        "--predict_wer", action="store_true", help="Use WER as the metric for best model instead of loss"
    )
    parser.add_argument("--max_eval_set_size", type=int, help="Maximum number of entries to fetch from eval dataset.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Per-device eval batch size.")

    parser.add_argument("--run_name", help="Run name to report to the run tracker")
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of step between each log")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.use_preprocessed and (args.train_datasets or args.eval_datasets):
        raise ValueError("Cannot use both preprocessed data and specify train/eval datasets. Choose one method.")

    if args.use_preprocessed and args.save_processed:
        raise ValueError("Cannot use preprocessed data and save preprocessed data at the same time.")

    processor = WhisperProcessor.from_pretrained(args.model_name, language="hebrew", task="transcribe")
    preparator = DatasetPreparator(
        processor,
        proc_num=args.ds_processor_proc_num,
        timestamp_sample_prob=args.include_timestamps_prob,
        condition_on_prev_sample_prob=args.include_prev_text_prob,
        inject_synthetic_timestamps=args.inject_synthetic_timestamps,
        audio_shift_augmentation=args.audio_shift_augmentation,
    )

    if args.use_preprocessed:
        preprocessed_dataset_dicts = []
        for preprocessed in args.use_preprocessed:
            try:
                # Try to load from disk first
                dataset_dict = load_from_disk(preprocessed)
            except FileNotFoundError:
                # If not found on disk, try to load as a remote dataset
                dataset_dict = load_dataset(preprocessed)
            preprocessed_dataset_dicts.append(dataset_dict)

        if len(preprocessed_dataset_dicts) == 1:
            train_set = dataset_dict["train"]
            eval_set = dataset_dict["eval"]
        else:
            probs = None
            if args.use_preprocessed_probs is not None:
                assert len(args.use_preprocessed_probs) == len(preprocessed_dataset_dicts)
                probs = args.use_preprocessed_probs
            train_set = interleave_datasets(
                [d["train"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # We set the seed so each distributed process will interleave in the same way
                # otherwise - the dataloader across each process ends up with different lengths
                # which screws up the collective synchronization
                # See https://huggingface.co/docs/accelerate/en/concept_guides/internal_mechanism
                seed=745,
            )
            eval_set = interleave_datasets(
                [d["eval"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # We set the seed so each distributed process will interleave in the same way
                # See above.
                seed=722,
            )

    elif args.save_processed:

        if not args.train_datasets or not args.eval_datasets:
            raise ValueError("Both --train_datasets and --eval_datasets must be provided when using --save_processed")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets(args.eval_datasets)

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

        dataset_dict = DatasetDict({"train": train_set, "eval": eval_set})
        dataset_dict.save_to_disk(args.save_processed)
        print(f"Preprocessed datasets saved to {args.save_processed}")
        return  # Exit after saving preprocessed data
    else:
        if not args.train_datasets or not args.eval_datasets:
            raise ValueError("Both --train_datasets and --eval_datasets must be provided for training")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets(args.eval_datasets)

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

    if args.max_eval_set_size:
        eval_set = eval_set.select(range(args.max_eval_set_size))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    )

    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    if args.use_qlora:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, attn_implementation=args.attn_implementation
        )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    assert (
        model.config.max_target_positions == whisper_max_target_positions
    ), f"Model max_target_positions {model.config.max_target_positions} != {whisper_max_target_positions}"

    if args.use_qlora:
        model = prepare_model_for_qlora(model)

    model.config.use_cache = False

    model.generate = partial(model.generate, language="hebrew", task="transcribe", use_cache=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,  # Overidden by warmup_steps - So cannot really use this?
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=model.config.max_target_positions,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="all" if args.run_name else "none",
        load_best_model_at_end=False,
        metric_for_best_model="wer" if args.predict_wer else "loss",
        greater_is_better=False,
        push_to_hub=(not args.skip_push_to_hub),
        run_name=args.run_name,
        hub_model_id=f"{args.hf_org_name}/{args.output_model_name}" if not args.skip_push_to_hub else None,
        remove_unused_columns=False,
        # Configure mixed precision based on the argument
        bf16=True if args.mixed_precision == "bf16" else None,
        fp16=True if args.mixed_precision == "fp16" else None,
        tf32=True if args.mixed_precision == "tf32" else None,
        # Configure prediction loss and metric based on predict_wer
        prediction_loss_only=False if args.predict_wer else True,
        # Configure save_total_limit if max_checkpoints_to_keep is provided
        save_total_limit=args.max_checkpoints_to_keep,
        # Configure save_only_model
        save_only_model=True if args.save_only_model else None,

        # There is not branching in training the Whisper model
        ddp_find_unused_parameters=False,
        # This would take longer, but will calculate the loss
        # with proper averaging across GPUs.
        # this is important when the dataset samples vary
        # wildly in the amount of tokens contributing to the loss and
        # the distribution of those samples is very unbalanced
        average_tokens_across_devices=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer),
        processing_class=processor,
        compute_loss_func=compute_loss_func,
    )

    resume_from_checkpoint = False
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        resume_from_checkpoint = True
        if args.resume_from_checkpoint_path is not None:
            resume_from_checkpoint = args.resume_from_checkpoint_path
            print(f"Resuming checkpoint {resume_from_checkpoint}")
        else:
            print("No checkpoint path provided, resuming from latest")

    print("Start training!")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the model
    trainer.save_model(args.output_model_name)


if __name__ == "__main__":
    main()
