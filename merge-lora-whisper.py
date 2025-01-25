#!/usr/bin/env python3
# coding: utf-8

import argparse
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with Whisper base model")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name or path of the base Whisper model")
    parser.add_argument("--lora_model_name", type=str, required=True, help="Name or path of the LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the merged model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="Model ID for pushing to Hugging Face Hub")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model_name}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model_name, device_map="auto", torch_dtype=torch.float16
    )

    print(f"Loading LoRA adapter: {args.lora_model_name}")
    lora_model = PeftModel.from_pretrained(
        base_model, args.lora_model_name, device_map="auto", torch_dtype=torch.float16
    )

    print("Merging weights")
    merged_model = lora_model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    print("Saving processor")
    processor = WhisperProcessor.from_pretrained(args.base_model_name)
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")

        print(f"Pushing merged model to Hugging Face Hub: {args.hub_model_id}")
        merged_model.push_to_hub(args.hub_model_id)
        processor.push_to_hub(args.hub_model_id)

    print("Done!")


if __name__ == "__main__":
    main()
