#!/usr/bin/env python3
# coding: utf-8

import argparse
import evaluate
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

from datasets import Audio, load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
from functools import partial

from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training


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
    parser.add_argument("--model_name", default="openai/whisper-large-v2", help="Name of the model to train")
    parser.add_argument("--output_model_name", required=True, help="Name of the fine-tuned model to generate")
    parser.add_argument(
        "--eval_dataset",
        help="Reference dataset for evaluation. Format: dataset_name[:split_name]:text_column",
    )
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between each model save/upload.")
    parser.add_argument("--max_eval_set_size", type=int, help="Maximum number of entries to fetch from eval dataset.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Per-device eval batch size.")

    return parser.parse_args()


def load_datasets(dataset_specs):
    datasets = []
    for spec in dataset_specs:
        parts = spec.split(":")
        dataset_name = parts[0]
        split = parts[1] if len(parts) > 2 else "train"
        text_column = parts[-1]
        dataset = load_dataset(dataset_name, split=split)
        datasets.append((dataset, text_column))
    return datasets


def prepare_dataset(example, processor, text_column_name):
    try:
        audio = example["audio"]
        original_sampling_rate = audio["sampling_rate"]
        target_sampling_rate = 16000

        if original_sampling_rate != target_sampling_rate:
            resampler = Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
            audio_array = torch.tensor(audio["array"]).float()
            resampled_audio_array = resampler(audio_array).numpy()
        else:
            resampled_audio_array = audio["array"]

        # Add a space.
        # Motivation: sometimes post-training models glue words together.
        text = f' {example[text_column_name]}'

        example = processor(
            audio=resampled_audio_array,
            sampling_rate=target_sampling_rate,
            text=text,
        )

        example["input_length"] = len(resampled_audio_array) / target_sampling_rate

        return example
    except Exception as e:
        print(f"Exception: {e}")
        return None

def select_legal_entries(processed_dataset):
    indices = []
    for idx, e in enumerate(processed_dataset['labels']):
        if len(e) <= 448:
            indices.append(idx)

    return processed_dataset.select(indices)

def process_datasets(datasets, processor):
    processed_datasets = []
    for dataset, text_column in datasets:
        prepare_dataset_func = lambda example: prepare_dataset(example, processor, text_column)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
        processed_dataset = dataset.map(prepare_dataset_func, remove_columns=dataset.column_names, num_proc=1)
        processed_dataset = select_legal_entries(processed_dataset) 
        processed_datasets.append(processed_dataset)
    return concatenate_datasets(processed_datasets) if len(processed_datasets) > 1 else processed_datasets[0]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor, metric, normalizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

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

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

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

        train_set = process_datasets(train_datasets, processor)
        eval_set = process_datasets(eval_datasets, processor)

        dataset_dict = DatasetDict({"train": train_set, "eval": eval_set})
        dataset_dict.save_to_disk(args.save_processed)
        print(f"Preprocessed datasets saved to {args.save_processed}")
        return  # Exit after saving preprocessed data
    else:
        if not args.train_datasets or not args.eval_dataset:
            raise ValueError("Both --train_datasets and --eval_dataset must be provided for training")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets([args.eval_dataset])

        train_set = process_datasets(train_datasets, processor)
        eval_set = process_datasets(eval_datasets, processor)

    if args.max_eval_set_size:
        eval_set = eval_set.select(range(args.max_eval_set_size))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

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
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=448,
        logging_strategy="steps",
        save_steps=500,
        report_to="all",
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        hub_model_id=f'ivrit-ai/{args.output_model_name}'
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer),
        tokenizer=processor,
    )

    print("Start training!")
    trainer.train()

    # Save the model
    trainer.save_model(args.output_model_name)


if __name__ == "__main__":
    main()
