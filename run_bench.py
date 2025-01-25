#!/usr/bin/env python3

import argparse
import os
import subprocess


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define dataset configurations as list of tuples
    datasets = [
        ("ivrit-ai/eval-d1:test:text", None, "ivrit_ai_eval_d1"),
        ("ivrit-ai/saspeech:test:text", None, "saspeech"),
        ("google/fleurs:test:transcription", "he_il", "fleurs"),
        ("mozilla-foundation/common_voice_17_0:test:sentence", "he", "common_voice_17"),
        ("imvladikon/hebrew_speech_kan:validation:sentence", None, "hebrew_speech_kan"),
    ]

    # Iterate over datasets and run evaluation
    for ds_path, ds_name, ds_output_name in datasets:
        output_file = os.path.join(args.output_dir, f"{ds_output_name}.csv")

        if os.path.exists(output_file) and not args.overwrite:
            print(f"Skipping {ds_path} - output file {output_file} already exists")
            continue

        print(f"Evaluating {ds_path}...")

        cmd = [
            "./evaluate_model.py",
            "--engine",
            args.engine,
            "--model",
            args.model,
            "--dataset",
            ds_path,
            "--workers",
            str(args.workers),
            "--output",
            output_file,
        ]

        if ds_name:
            cmd.extend(["--name", ds_name])

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
