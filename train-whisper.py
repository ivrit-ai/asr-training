#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk, ReadInstruction
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class FastEvalWhisperTrainer(Seq2SeqTrainer):
    """Runs generation only on the first `num_gen_samples` eval samples;
    every other sample just computes teacher-forced loss (fast)."""

    def __init__(self, num_gen_samples=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_gen_samples = num_gen_samples

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        remaining = self.num_gen_samples - getattr(self, "_gen_counter", 0)
        if remaining > 0:
            batch_size = next(
                v.shape[0] for v in inputs.values()
                if isinstance(v, torch.Tensor) or hasattr(v, "shape")
            )
            self._gen_counter = getattr(self, "_gen_counter", 0) + batch_size
            return super().prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
        return super().prediction_step(model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys)

    def evaluate(self, *args, **kwargs):
        self._gen_counter = 0
        return super().evaluate(*args, **kwargs)
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

        
        try:
            dataset = load_dataset(dataset_name, split=split)
            if dataset.builder_name == "json" and not "transcript" in dataset.features:
                print(f"Assumed dataset format mis-detection. Attempting to load. using `load_from_disk` instead. (See comments in code)")
                raise ValueError("Dataset format mis-detection.")
        
        # Local datasets, could suffer from a bug where there are more ".json" files
        # than ".arrow" files which leads to a mis-detection of the dataset format.
        # The "load_from_disk" API can get around this problem since it's designed to load
        # such locally stored dataset generated using "save_to_disk"
        except:
            dataset = load_from_disk(dataset_name)
            
            # But, we want to support the flexible "split instruction" syntax like load_dataset provides.
            # Hf made this extremely hard, by hiding the parsing and results inside a wrapped internal class.
            # Why? why HF ?!
            read_instruction = ReadInstruction.from_spec(split)
            actual_ri_data = read_instruction._relative_instructions[0]
            slice_units = actual_ri_data.unit
            # We won't go that crazy - only support "abs" units (not pct syntax)
            if slice_units != 'abs':
                # This is such shame - HF please fix this.
                raise ValueError(f'Unable to support the split definition: ${split} - please read the code for more details.')
            
            split_name = actual_ri_data.splitname
            from_entry = actual_ri_data.from_
            to_entry = actual_ri_data.to
            dataset = dataset[split_name]
            if from_entry is not None:
                dataset = dataset.skip(from_entry)
            else:
                from_entry = 0
            if to_entry is not None:
                dataset = dataset.take(to_entry - from_entry)

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


# Whisper's language/task/timestamp control tokens (e.g. <|he|>, <|transcribe|>,
# <|notimestamps|>) can survive tokenizer.batch_decode(skip_special_tokens=True)
# since some are tied to the previous-text-prompt tokens/timestamp tokens which
# aren't always flagged as "special" by the tokenizer. Strip them explicitly so
# ref/hyp shown for eyeballing (console + wandb table) are an exact text comparison.
special_token_display_pattern = re.compile(r"<\|[^|>]*\|>")


def strip_special_tokens_for_display(text):
    return special_token_display_pattern.sub("", text).strip()


def compute_metrics(pred, processor, metric, normalizer, trainer_ref=None):
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

    # Always echo a few ref/hyp pairs to stdout so they land in the captured
    # console log even if the wandb Table logging below fails/gets dropped.
    num_samples_to_show = min(5, len(pred_str))
    if num_samples_to_show > 0:
        print(f"[eval_samples] showing {num_samples_to_show} ref/hyp sample(s):")
        for i in range(num_samples_to_show):
            ref_display = strip_special_tokens_for_display(label_str[i])
            hyp_display = strip_special_tokens_for_display(pred_str[i])
            print(f"[eval_samples]   ref[{i}]: {ref_display!r}")
            print(f"[eval_samples]   hyp[{i}]: {hyp_display!r}")

    try:
        import wandb
        if wandb.run is not None and len(pred_str) > 0:
            sample_table = wandb.Table(columns=["ref", "hyp"])
            for i in range(num_samples_to_show):
                sample_table.add_data(
                    strip_special_tokens_for_display(label_str[i]),
                    strip_special_tokens_for_display(pred_str[i]),
                )

            # IMPORTANT: wandb.log() without an explicit `step=` uses its own
            # internal auto-incrementing step counter, which is SEPARATE from
            # the step counter the HF Trainer's WandbCallback uses (it logs
            # scalars with an explicit step=state.global_step, jumping by
            # eval_steps/logging_steps each time). Left unstepped, this call
            # only advances by +1 per eval invocation, so it quickly falls
            # behind the explicit-step timeline. Per wandb docs, a run can
            # only write to the "current" and "next" step - it cannot write
            # backward - so once our counter falls behind, every subsequent
            # call here is silently dropped and the table stops updating.
            # Pinning to the trainer's real global_step keeps both timelines
            # in sync so the table keeps recording for the full run.
            log_kwargs = {}
            trainer = trainer_ref.get("trainer") if trainer_ref is not None else None
            if trainer is not None:
                log_kwargs["step"] = trainer.state.global_step

            wandb.log({
                "eval_samples": sample_table,
            }, **log_kwargs)
    except ImportError:
        pass

    return {"wer_ortho": wer_ortho, "wer": wer}


def freeze_encoder(model):
    """Freeze all audio-encoder parameters (they stay out of the optimizer)."""
    frozen = 0
    for p in model.model.encoder.parameters():
        if p.requires_grad:
            p.requires_grad = False
            frozen += p.numel()
    print(f"[freeze] encoder frozen ({frozen:,} params will not train)")


def freeze_decoder_except_embeddings(model):
    """
    Freeze every decoder parameter EXCEPT the token embeddings (embed_tokens, tied to
    proj_out). Applied BEFORE the trainer/DDP wrap, so the frozen params are excluded
    from DDP's reducer entirely — this is DDP-safe (unlike a mid-run requires_grad flip).

    Use this for "train embeddings only" runs (the static equivalent of an effectively
    infinite embedding warmup).
    """
    embed_ids = {id(p) for p in model.model.decoder.embed_tokens.parameters()}
    frozen = 0
    for _, p in model.model.decoder.named_parameters():
        if id(p) in embed_ids:
            continue  # keep token embeddings trainable
        if p.requires_grad:
            p.requires_grad = False
            frozen += p.numel()
    print(f"[freeze] decoder frozen except embeddings ({frozen:,} params will not train)")


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


class EmbeddingWarmupCallback(TrainerCallback):
    """
    Stage the decoder unfreeze: for the first `warmup_steps` optimizer steps, train ONLY
    the decoder token embeddings (embed_tokens, tied to proj_out); then release the rest
    of the decoder so the whole decoder trains.

    How it stays correct with the HF Trainer optimizer:
      - The optimizer is built once, over params that require grad AT THAT TIME. So the
        rest-of-decoder params must be trainable when the optimizer is created (they are:
        we only flip them off in on_train_begin, which runs AFTER optimizer creation).
      - During warmup those params have requires_grad=False -> no grad is produced ->
        AdamW skips them entirely (no update, no weight-decay leak).
      - At `warmup_steps` we flip them back on and they start updating.

    NOTE (multi-GPU): DDP registers grad hooks at construction over the then-trainable
    params. Flipping requires_grad mid-run can break DDP. For DDP, prefer running the
    warmup as a separate short job, or keep this to single-process training.
    """

    def __init__(self, model, warmup_steps: int):
        self.model = model
        self.warmup_steps = warmup_steps
        self._frozen_params = []
        self._released = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.warmup_steps <= 0:
            return
        embed = self.model.model.decoder.embed_tokens
        embed_ids = {id(p) for p in embed.parameters()}
        for _, p in self.model.model.decoder.named_parameters():
            if id(p) in embed_ids:
                continue  # keep embeddings trainable
            if p.requires_grad:
                p.requires_grad = False
                self._frozen_params.append(p)
        print(
            f"[warmup] embeddings-only for {self.warmup_steps} steps "
            f"(temporarily froze {len(self._frozen_params)} decoder tensors)"
        )

    def on_step_begin(self, args, state, control, **kwargs):
        if self.warmup_steps > 0 and not self._released and state.global_step >= self.warmup_steps:
            for p in self._frozen_params:
                p.requires_grad = True
            self._released = True
            print(f"[warmup] released decoder at step {state.global_step}; full decoder now trains")


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
    parser.add_argument(
        "--target_language", type=str, default="hebrew", help="The target training language (Only a single language training is currently supported)"
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
    parser.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the whole audio encoder (default: off)."
        "For tokenizer-only optimization the acoustics are unchanged, so the encoder should stay frozen.",
    )
    parser.add_argument(
        "--embedding_warmup_steps",
        type=int,
        default=0,
        help="Train ONLY the decoder token embeddings for this many optimizer steps before "
        "releasing the rest of the decoder. 0 disables the warmup (full decoder from step 0). "
        "NOTE: this staged unfreeze uses a mid-run requires_grad flip and is SINGLE-PROCESS ONLY "
        "(it breaks DDP). Under DDP use --train_embeddings_only, or --ddp_find_unused_parameters.",
    )
    parser.add_argument(
        "--train_embeddings_only",
        action="store_true",
        help="Static, DDP-safe: freeze everything except the decoder token embeddings BEFORE the "
        "trainer/DDP wrap (implies encoder frozen). Use this instead of a very long "
        "--embedding_warmup_steps when training with DDP/accelerate.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set DDP find_unused_parameters=True. Needed if some trainable params don't receive "
        "grad every step (e.g. staged --embedding_warmup_steps under DDP). Adds overhead.",
    )
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
        "--predict_wer", action="store_true", default=False, help="Predict WER for all eval samples and report metrics. Implies prediction_loss_only=False."
    )
    parser.add_argument(
        "--eval_wer_sample_size", type=int, default=0, help="Number of eval samples to run generation + WER on (via FastEvalWhisperTrainer). 0 disables (uses Seq2SeqTrainer)."
    )
    parser.add_argument(
        "--eval_on_first_step", action="store_true", default=False, help="Run evaluation at step 1 (EvaluateFirstStepCallback)."
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

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.target_language, task="transcribe")
    preparator = DatasetPreparator(
        processor,
        proc_num=args.ds_processor_proc_num,
        timestamp_sample_prob=args.include_timestamps_prob,
        condition_on_prev_sample_prob=args.include_prev_text_prob,
        inject_synthetic_timestamps=args.inject_synthetic_timestamps,
        audio_shift_augmentation=args.audio_shift_augmentation,
    )

    dataset_shuffle_seed = 745
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
                seed=dataset_shuffle_seed,
            )
            eval_set = interleave_datasets(
                [d["eval"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # We set the seed so each distributed process will interleave in the same way
                # See above.
                seed=dataset_shuffle_seed,
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
        eval_set = eval_set.shuffle(seed=dataset_shuffle_seed).select(range(args.max_eval_set_size))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    )

    metric = evaluate.load("wer", experiment_id=f"rank{os.environ.get('RANK', '0')}")
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

    # --- Freezing controls (tokenizer-optimization training) ---
    # For tokenizer-only optimization the acoustics don't change, so the encoder should
    # stay frozen; the decoder (esp. the new token embeddings) is what adapts.
    #
    # DDP note: any param that is trainable when the model is wrapped for DDP must receive
    # a gradient every step (unless find_unused_parameters=True). So all freezing that
    # should hold under DDP MUST happen HERE, before the trainer wraps the model — not in
    # a callback. --train_embeddings_only does exactly that; --embedding_warmup_steps flips
    # requires_grad mid-run and is single-process only.
    if args.use_qlora:
        if args.freeze_encoder or args.embedding_warmup_steps > 0 or args.train_embeddings_only:
            print("[freeze] freezing flags are ignored with --use_qlora")
    elif args.train_embeddings_only:
        # Static, DDP-safe: encoder + decoder(except embeddings) frozen up front.
        freeze_encoder(model)
        freeze_decoder_except_embeddings(model)
    elif args.freeze_encoder:
        freeze_encoder(model)

    model.generate = partial(model.generate, language=args.target_language, task="transcribe", use_cache=True)

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
        prediction_loss_only=not (args.predict_wer or args.eval_wer_sample_size > 0),
        # Configure save_total_limit if max_checkpoints_to_keep is provided
        save_total_limit=args.max_checkpoints_to_keep,
        # Configure save_only_model
        save_only_model=True if args.save_only_model else None,
        # There is not branching in training the Whisper model
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        # This would take longer, but will calculate the loss
        # with proper averaging across GPUs.
        # this is important when the dataset samples vary
        # wildly in the amount of tokens contributing to the loss and
        # the distribution of those samples is very unbalanced
        average_tokens_across_devices=True,
    )

    # Holder used so compute_metrics can look up the trainer's current
    # global_step (needed to keep wandb table logging on the same step
    # timeline as the trainer's own scalar metric logging - see comment
    # in compute_metrics for why this matters). Populated right after the
    # trainer is constructed below.
    trainer_ref = {}

    if args.eval_wer_sample_size > 0:
        trainer = FastEvalWhisperTrainer(
            num_gen_samples=args.eval_wer_sample_size,
            args=training_args,
            model=model,
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer, trainer_ref),
            processing_class=processor,
            compute_loss_func=compute_loss_func,
        )
    else:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer, trainer_ref),
            processing_class=processor,
            compute_loss_func=compute_loss_func,
        )
    trainer_ref["trainer"] = trainer

    # Staged decoder unfreeze: train embeddings only, then release the rest of the decoder.
    # This flips requires_grad mid-run, which is incompatible with DDP unless
    # find_unused_parameters=True. Warn if that looks likely to break.
    if not args.use_qlora and args.embedding_warmup_steps > 0:
        if args.train_embeddings_only:
            print("[warmup] --train_embeddings_only is set; ignoring --embedding_warmup_steps "
                  "(nothing to release — decoder stays frozen).")
        else:
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
                    and not args.ddp_find_unused_parameters:
                print("[warmup] WARNING: --embedding_warmup_steps under DDP will fail unless "
                      "--ddp_find_unused_parameters is set. Prefer --train_embeddings_only.")
            trainer.add_callback(EmbeddingWarmupCallback(model, args.embedding_warmup_steps))

    if args.eval_on_first_step:
        trainer.add_callback(EvaluateFirstStepCallback())

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
