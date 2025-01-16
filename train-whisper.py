#!/usr/bin/env python3
# coding: utf-8

import argparse
import evaluate
import numpy as np
import pandas as pd
import re
import torch
import torchaudio
from torchaudio.transforms import Resample

from datasets import (
    Audio,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from functools import partial

from transformers import WhisperProcessor, BatchFeature
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from peft import prepare_model_for_kbit_training, PeftModel, LoraModel, LoraConfig, get_peft_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Whisper model with custom datasets.")
    parser.add_argument(
        "--train_datasets",
        nargs="*",
        help="Dataset(s) to train on. Format: dataset_name[:split_name]:text_column",
    )
    parser.add_argument("--save_processed", help="Dataset name to save processed data (will save both train and eval)")
    parser.add_argument(
        "--use_preprocessed", help="Dataset name to load preprocessed data from (either local path or remote dataset)"
    )
    parser.add_argument(
        "--ds_processor_proc_num", type=int, default=1, help="Number of parallel processors for datasets preparation"
    )
    parser.add_argument("--model_name", default="openai/whisper-large-v2", help="Name of the model to train")
    parser.add_argument("--output_model_name", required=True, help="Name of the fine-tuned model to generate")
    parser.add_argument("--hf_org_name", default="ivrit-ai", help="Name of HF Org to push the model to")
    parser.add_argument("--skip_push_to_hub", action="store_true", help="Don't push result model to hub")
    parser.add_argument(
        "--eval_dataset",
        help="Reference dataset for evaluation. Format: dataset_name[:split_name]:text_column",
    )
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Try and resue for last saved checkpoint")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--eval_steps", type=int, help="Number of steps between two evals, if not specified defaults to logging_steps."
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between each model save/upload.")
    parser.add_argument("--max_eval_set_size", type=int, help="Maximum number of entries to fetch from eval dataset.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Per-device eval batch size.")

    parser.add_argument("--run_name", help="Run name to report to the run tracker")
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of step between each log")

    return parser.parse_args()


# Split on : but allow : inside [] for the HF split slicing syntax
# https://huggingface.co/docs/datasets/loading#slice-splits
dataset_spec_split_pattern = r":(?=(?:[^\[\]]|\[[^\[\]]*\])*$)"


def load_datasets(dataset_specs):
    datasets = []
    for spec in dataset_specs:
        spec_version = None
        dataset_format_version = None
        text_column = None
        if spec.endswith("#v2"):
            spec_version = "v2"
            dataset_format_version = "v2"
            spec = spec[:-3]

        parts = re.split(dataset_spec_split_pattern, spec)

        dataset_name = parts[0]
        if spec_version == "v2":
            split = parts[1]
        else:
            split = parts[1] if len(parts) > 2 else "train"
            text_column = parts[-1]

        dataset = load_dataset(dataset_name, split=split)
        datasets.append((dataset, text_column, dataset_format_version))
    return datasets


# This is defined as part of the model config
# and should match the loaded model.
# For all whisper models so far this is the same value
# Ideally we will take this from the WhisperConfig but we need to
# Process the dataset before loading the model in some use cases.
whisper_max_target_positions = 448


class DatasetPreparator:
    def __init__(
        self,
        processor: WhisperProcessor,
        # Don't change this for Whisper
        tokenizer_time_precision=0.02,
        timestamp_sample_prob=0.5,
        condition_on_prev_sample_prob=0.5,
        seed: np.random.RandomState = None,
        proc_num: int = 1,
        device: str = "cpu",
    ):
        if proc_num > 1:  # Parallel processing will not work in multi threaded env.
            torch.set_num_threads(1)

        # Stability of the seed will allow reusing cached processing
        self.seed = np.random.default_rng(998) if seed is None else seed
        self.device = device
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.proc_num = proc_num
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.tokenizer_time_precision = tokenizer_time_precision

        orig_predict_timestamps = self.processor.tokenizer.predict_timestamps
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=False)
        self.prefix_tokens_no_ts = self.processor.tokenizer.prefix_tokens
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=True)
        self.prefix_tokens_with_ts = self.processor.tokenizer.prefix_tokens
        self.processor.tokenizer.set_prefix_tokens(predict_timestamps=orig_predict_timestamps)

        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.start_of_prev_token_id = self.tokenizer.convert_tokens_to_ids("<|startofprev|>")
        self.no_timestamp_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.timestamp_begin_token_id = self.no_timestamp_token_id + 1
        self.last_timestamp_token = self.tokenizer.total_vocab_size - 1
        self.total_timestamp_tokens = self.last_timestamp_token - self.timestamp_begin_token_id + 1
        self.max_allowed_tokenized_timestamp = (self.total_timestamp_tokens - 1) * self.tokenizer_time_precision
        self.prev_ids_max_length = whisper_max_target_positions // 2
        self.timestamp_sample_prob = timestamp_sample_prob
        self.condition_on_prev_sample_prob = condition_on_prev_sample_prob

        # Prepare the output features - to ensure optimal storage during mapping (uses disk cache for mapped content)
        self.output_features = Features(
            {
                "input_features": Sequence(feature=Sequence(feature=Value(dtype="float32"))),
                "labels": Sequence(feature=Value(dtype="int32")),
                "pad_value": Sequence(feature=Value(dtype="float32")),
                "pad_amount": Value("int32"),
                "input_length": Value("float64"),
            }
        )

    def _get_token_timestamp_for_time(self, time: float) -> int:
        # Sanity - time cannot be more than max timestamp token.
        if self.max_allowed_tokenized_timestamp < time:
            raise ValueError(f"Time {time} is too large. Max allowed time is {self.max_allowed_tokenized_timestamp}")

        # Round to nearest multiple of timestamp_token_time_precision
        return self.timestamp_begin_token_id + int(round(time / self.tokenizer_time_precision))

    def _select_legal_entries(self, processed_dataset):
        return processed_dataset.filter(
            lambda labels: len(labels) <= whisper_max_target_positions, input_columns="labels"
        )

    def _prepare_example_audio(self, is_batched, example, result_example: BatchFeature) -> None:
        audio_batched = example["audio"]
        if not is_batched:
            audio_batched = [audio_batched]

        resampled_audio = []
        for audio in audio_batched:
            original_sampling_rate = audio["sampling_rate"]
            target_sampling_rate = self.target_sampling_rate

            if original_sampling_rate != target_sampling_rate:
                resampler = Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
                audio_array = torch.tensor(audio["array"]).float()
                resampled_audio_array = resampler(audio_array).numpy()
            else:
                resampled_audio_array = audio["array"]

            resampled_audio.append(resampled_audio_array)

        # We want to use the device kwargs - we call the feature extractor directly
        # to avoid warning from the tokenizer (which does not know how to consume a device kwarg)
        feature_extraction_result = self.processor.feature_extractor(
            resampled_audio, sampling_rate=target_sampling_rate, return_attention_mask=True, device=self.device
        )

        input_feat = feature_extraction_result["input_features"]
        attn_mask = feature_extraction_result["attention_mask"]

        input_features_to_keep = []
        pad_values = []
        pad_amounts = []
        for input_feat_sample, attn_mask_sample in zip(input_feat, attn_mask):
            # Only makes sense to compress when we have at least 2 paddings to replace with 1
            if attn_mask_sample[-3] == 0:
                # +1 because the STFT window overflows slightly into the first silence padding frame
                # And the model might learned to use that signal to recognize start of silence.
                padding_starts_at_index = attn_mask_sample.argmin() + 1
                padding_vals = input_feat_sample.T[padding_starts_at_index]
                pad_values.append(padding_vals)
                input_feat_keep = input_feat_sample.T[:padding_starts_at_index].T
                input_features_to_keep.append(input_feat_keep)
                orig_out_features_length = input_feat_sample.shape[-1]
                pad_amounts.append(np.int32(orig_out_features_length - padding_starts_at_index))
            else:
                input_features_to_keep.append(input_feat_sample)
                pad_values.append(np.array([], dtype=input_feat_sample.dtype))
                pad_amounts.append(np.int32(0))

        if not is_batched:
            input_features_to_keep = input_features_to_keep[0]
            pad_values = pad_values[0]
            pad_amounts = pad_amounts[0]
        result_example["input_features"] = input_features_to_keep
        result_example["pad_value"] = pad_values
        result_example["pad_amount"] = pad_amounts

        input_lengths = [len(resampled_audio_array) / target_sampling_rate for resampled_audio_array in resampled_audio]
        if not is_batched:
            input_lengths = input_lengths[0]
        result_example["input_length"] = input_lengths

    def _prepare_example_text(self, is_batched, example, result_example, text_column_name, format_version) -> None:
        # Currently v2 format contains timestampped data and the prev transcript to condition on
        is_v2_format = format_version == "v2"
        can_add_timestamps = False
        prev_text_column_name = None
        tokenizer_kwargs = dict(
            add_special_tokens=True,
            # Motivation: sometimes post-training models glue words together.
            add_prefix_space=True,
            return_attention_mask=False,
        )
        if is_v2_format:
            text_column_name = "transcript"
            prev_text_column_name = "prev_transcript"
            tokenizer_kwargs["add_special_tokens"] = False
            tokenizer_kwargs["add_prefix_space"] = False  # v2 preformatted the text

        text = example[text_column_name]  # Text may contain timestamps

        tokenizer_result = self.processor.tokenizer(
            text,
            **tokenizer_kwargs,
        )

        if is_v2_format:
            can_add_timestamps = True  # For now assume v2 format always has timestamps
            result_example_labels = []
            token_ids_batch = tokenizer_result["input_ids"]
            prev_text_batch = example[prev_text_column_name]
            has_prev_batch = example["has_prev"]

            if not is_batched:
                token_ids_batch = [token_ids_batch]
                prev_text_batch = [prev_text_batch]
                has_prev_batch = [has_prev_batch]

            for token_ids, prev_text, has_prev in zip(token_ids_batch, prev_text_batch, has_prev_batch):

                should_train_on_timestamps = bool(self.seed.binomial(1, self.timestamp_sample_prob))
                should_condition_on_prev = bool(self.seed.binomial(1, self.condition_on_prev_sample_prob))

                if can_add_timestamps and not should_train_on_timestamps:
                    # Remove all timestamp tokens
                    token_ids = [token_id for token_id in token_ids if token_id < self.timestamp_begin_token_id]
                    # Note - no-timestamp token id is prepended as part of the prefix later.

                prev_ids = []
                if should_condition_on_prev and has_prev:
                    prev_ids = self.processor.tokenizer(
                        prev_text,
                        **tokenizer_kwargs,
                    )["input_ids"]

                    if not should_train_on_timestamps:
                        prev_ids = [token_id for token_id in prev_ids if token_id < self.timestamp_begin_token_id]

                    # Calculate how manh prev_ids we want to take
                    max_prev_ids_len_to_take = min(
                        whisper_max_target_positions - len(token_ids)
                        # 3 - Prefix for transcription (sot+lang+task)
                        # 1 - eot token
                        # 1 - Prefix for prev (prev)
                        - 5,
                        # And anyway no more than half the max size
                        self.prev_ids_max_length,
                    )

                    # Take as much as we can from the prev_ids
                    prev_ids = prev_ids[-max_prev_ids_len_to_take:]

                    # prepend a prev token
                    prev_ids = [self.start_of_prev_token_id] + prev_ids

                with_timestamps = can_add_timestamps and should_train_on_timestamps
                prefix_tokens = self.prefix_tokens_with_ts if with_timestamps else self.prefix_tokens_no_ts
                all_input_ids = prev_ids + prefix_tokens + token_ids + [self.eot_token_id]

                result_example_labels.append(all_input_ids)

            if not is_batched:
                result_example_labels = result_example_labels[0]

            result_example["labels"] = result_example_labels
        else:
            result_example["labels"] = tokenizer_result["input_ids"]

    def _prepare_example_fn(self, example, text_column_name, format_version):
        is_batched = isinstance(example["audio"], (list, tuple))
        try:
            # Ensure proper naming of the output features
            # As if we called the processoer with an audio param.
            # This is a very weird API quirk, but it's what the processor produces so we need to adapt.
            result_example = BatchFeature({})

            self._prepare_example_audio(is_batched, example, result_example)
            self._prepare_example_text(is_batched, example, result_example, text_column_name, format_version)

            return result_example
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def prepare_dataset(self, dataset: Dataset, text_column_name, format_version: str | None, batch_size: int = 1):

        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))

        batch_kwargs = {"batched": batch_size > 1, "batch_size": batch_size}

        columns_to_remove = dataset.column_names
        # If a DatasetDict was passed in, it contains multiple splits.
        # Take the names of the columns from any of the splits
        if isinstance(dataset, DatasetDict):
            any_split_name = next(iter(columns_to_remove.keys()))
            columns_to_remove = dataset[any_split_name].column_names

        processed_dataset = dataset.map(
            lambda x: self._prepare_example_fn(x, text_column_name, format_version),
            remove_columns=columns_to_remove,
            num_proc=self.proc_num,
            features=self.output_features,
            **batch_kwargs,
        )

        processed_dataset = self._select_legal_entries(processed_dataset)

        return processed_dataset


def process_datasets(datasets, preparator: DatasetPreparator):
    processed_datasets = [
        preparator.prepare_dataset(dataset=dataset, text_column_name=text_column, format_version=dataset_format_version)
        for dataset, text_column, dataset_format_version in datasets
    ]
    return concatenate_datasets(processed_datasets) if len(processed_datasets) > 1 else processed_datasets[0]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

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

        batch = BatchFeature({"input_features": torch.stack(input_features)})

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"]

        # Labels, represent the input to the decoder
        batch["decoder_input_ids"] = labels[:, :-1]

        # Where we do not need to attend when calculating loss - -100 is the agreed
        # ignored value for the pytorch loss functions
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Shift all labels to the left, thus the expected generated label
        # is at the same index of the generated output id from the decoder
        # and the loss function would compare them (cross entropy loss in this case)
        # Note - this means the is no loss calculated ot the first "start of transcript" token id
        # since it is not expected to be predicted but always provided.
        # The loss is calculated for the task/lang/notimestamp tokens since the model needs to know
        # to associate them with the proper output
        labels = labels[:, 1:]

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


def main():
    args = parse_arguments()

    if args.use_preprocessed and (args.train_datasets or args.eval_dataset):
        raise ValueError("Cannot use both preprocessed data and specify train/eval datasets. Choose one method.")

    if args.use_preprocessed and args.save_processed:
        raise ValueError("Cannot use preprocessed data and save preprocessed data at the same time.")

    processor = WhisperProcessor.from_pretrained(args.model_name, language="hebrew", task="transcribe")
    preparator = DatasetPreparator(processor, proc_num=args.ds_processor_proc_num)

    if args.use_preprocessed:
        try:
            # Try to load from disk first
            dataset_dict = load_from_disk(args.use_preprocessed)
        except FileNotFoundError:
            # If not found on disk, try to load as a remote dataset
            dataset_dict = load_dataset(args.use_preprocessed)

        train_set = dataset_dict["train"]
        eval_set = dataset_dict["eval"]
    elif args.save_processed:

        if not args.train_datasets or not args.eval_dataset:
            raise ValueError("Both --train_datasets and --eval_dataset must be provided when using --save_processed")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets([args.eval_dataset])

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

        dataset_dict = DatasetDict({"train": train_set, "eval": eval_set})
        dataset_dict.save_to_disk(args.save_processed)
        print(f"Preprocessed datasets saved to {args.save_processed}")
        return  # Exit after saving preprocessed data
    else:
        if not args.train_datasets or not args.eval_dataset:
            raise ValueError("Both --train_datasets and --eval_dataset must be provided for training")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets([args.eval_dataset])

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

    if args.max_eval_set_size:
        eval_set = eval_set.select(range(args.max_eval_set_size))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    if args.use_qlora:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
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
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,  # Overidden by warmup_steps - So cannot really use this?
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=model.config.max_target_positions,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=500,
        report_to="all",
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=(not args.skip_push_to_hub),
        run_name=args.run_name,
        hub_model_id=f"{args.hf_org_name}/{args.output_model_name}",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer),
        processing_class=processor,
    )

    print("Start training!")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the model
    trainer.save_model(args.output_model_name)


if __name__ == "__main__":
    main()
