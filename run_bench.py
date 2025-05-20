#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError


def extract_hf_model_path(ds_path: str) -> str:
    return ds_path.split(":")[0]


def is_dataset_gated(dataset_name: str) -> bool:
    """Check if a Hugging Face dataset is gated."""
    api = HfApi()
    try:
        api.auth_check(
            repo_id=extract_hf_model_path(dataset_name),
            repo_type="dataset",
            # Don't use the logged in user token to detect if this is a gated ds
            # regardless of this user access
            token=False,
        )
        return False  # If auth_check succeeds, the dataset is not gated
    except (RepositoryNotFoundError, GatedRepoError) as e:
        if isinstance(e, GatedRepoError):
            return True  # Dataset exists but is gated
        return False  # Dataset doesn't exist or is private


def check_access(dataset_name: str) -> bool:
    """Return True if the dataset is accessible (passes auth check), False otherwise.
    This function never throws any exceptions."""
    try:
        api = HfApi()
        api.auth_check(repo_id=extract_hf_model_path(dataset_name), repo_type="dataset")
        return True
    except Exception:
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to engine script (e.g. engines/faster_whisper_engine.py)")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation results")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if exists")
    parser.add_argument("--device", type=str, default="auto", help="copmute device")
    args = parser.parse_args()

    # Ensure engine script exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine script not found: {args.engine}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define dataset configurations as list of tuples
    datasets = [
        ("ivrit-ai/eval-d1:test:text", None, "ivrit_ai_eval_d1"),
        ("ivrit-ai/eval-whatsapp:test:text", None, "ivrit_ai_eval_whatsapp"),
        ("upai-inc/saspeech:test:text", None, "saspeech"),
        ("google/fleurs:test:transcription", "he_il", "fleurs"),
        ("mozilla-foundation/common_voice_17_0:test:sentence", "he", "common_voice_17"),
        ("imvladikon/hebrew_speech_kan:validation:sentence", None, "hebrew_speech_kan"),
    ]

    # Iterate over datasets and run evaluation
    for ds_path, ds_name, ds_output_name in datasets:
        if not check_access(ds_path):
            gated_msg = " (this is a gated dataset)" if is_dataset_gated(ds_path) else ""
            print(
                f"Warning: Dataset '{ds_path}' is not accessible{gated_msg}. Ensure you are logged in to HF and have permission to access it."
            )
            continue

        output_file = os.path.join(args.output_dir, f"{ds_output_name}.csv")

        print(f"Evaluating {ds_path}...")

        cmd = [
            "./evaluate_model.py",
            "--engine",
            str(engine_path.absolute()),
            "--model",
            args.model,
            "--dataset",
            ds_path,
            "--workers",
            str(args.workers),
            "--output",
            output_file,
            "--device",
            args.device
        ]

        if args.overwrite:
            cmd.append("--overwrite")
        if ds_name:
            cmd.extend(["--name", ds_name])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {ds_path}: {e}")
            continue


if __name__ == "__main__":
    main()
