#!/usr/bin/env python3

import argparse
import concurrent.futures
import os

import datasets
import jiwer
import pandas
import whisper.normalizers
from hebrew import Hebrew
from tqdm import tqdm


def clean_some_unicode_from_text(text):
    chars_to_remove = "\u061C"  # Arabic letter mark
    chars_to_remove += "\u200B\u200C\u200D"  # Zero-width space, non-joiner, joiner
    chars_to_remove += "\u200E\u200F"  # LTR and RTL marks
    chars_to_remove += "\u202A\u202B\u202C\u202D\u202E"  # LTR/RTL embedding, pop, override
    chars_to_remove += "\u2066\u2067\u2068\u2069"  # Isolate controls
    chars_to_remove += "\uFEFF"  # Zero-width no-break space
    return text.translate({ord(c): None for c in chars_to_remove})


def remove_niqqud(text: str):
    """Remove niqqud from Hebrew text."""
    return Hebrew(text).no_niqqud().string


class HebrewTextNormalizer:
    def __init__(self):
        self.whisper_normalizer = whisper.normalizers.BasicTextNormalizer()

    def __call__(self, text):
        text = clean_some_unicode_from_text(text)
        text = remove_niqqud(text)
        text = text.replace('"', "").replace("'", "")

        return self.whisper_normalizer(text)


def process_entry(args):
    i, entry, transcribe_fn, text_column, normalizer = args

    raw_ref_text = entry[text_column]
    raw_eval_text = transcribe_fn(entry)

    ref_text = normalizer(raw_ref_text)
    eval_text = normalizer(raw_eval_text)

    entry_metrics = jiwer.process_words([ref_text], [eval_text])

    entry_data = {
        "id": i,
        "reference_text": raw_ref_text,
        "predicted_text": raw_eval_text,
        "norm_reference_text": ref_text,
        "norm_predicted_text": eval_text,
        "wer": entry_metrics.wer,
        "wil": entry_metrics.wil,
        "substitutions": entry_metrics.substitutions,
        "deletions": entry_metrics.deletions,
        "insertions": entry_metrics.insertions,
        "hits": entry_metrics.hits,
    }

    for key in entry.keys():
        if key not in ["audio", text_column]:
            entry_data[f"metadata_{key}"] = entry[key]

    return entry_data


def calculate_final_metrics(df: pandas.DataFrame):
    df = df.sort_values(by=["id"])
    df["reference_text"] = df["reference_text"].fillna("")
    df["predicted_text"] = df["predicted_text"].fillna("")

    # convert to list of dicts
    entries_data = df.to_dict(orient="records")

    htn = HebrewTextNormalizer()

    # Calculate final metrics
    results = jiwer.process_words(
        [htn(entry["reference_text"]) for entry in entries_data],
        [htn(entry["predicted_text"]) for entry in entries_data],
    )

    return results

def evaluate_model(transcribe_fn, ds, text_column, num_workers=1):
    normalizer = HebrewTextNormalizer()
    entries_data = []

    # Prepare arguments for parallel processing
    process_args = [(i, ds[i], transcribe_fn, text_column, normalizer) for i in range(len(ds))]

    # Process entries in parallel with progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = [executor.submit(process_entry, arg) for arg in process_args]

        # Use tqdm to track progress
        entries_data = []
        with tqdm(total=len(futures), desc="Processing entries") as pbar:
            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                entries_data.append(entry)
                last_wer = entry["wer"]
                last_wil = entry["wil"]
                pbar.set_postfix(last_wer=f"{last_wer:.5f}", last_wil=f"{last_wil:.5f}")
                pbar.update(1)

    # Sort results by ID to maintain original order
    entries_data.sort(key=lambda x: x["id"])

    return pandas.DataFrame(entries_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a speech-to-text model.")
    parser.add_argument("--engine", type=str, required=True, help="Path to engine script")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to evaluate in format dataset_name:<split>:<text_column>"
    )
    parser.add_argument("--name", type=str, required=False, help="Optional name parameter for dataset.load_dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers to use for evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite exists outputs, otherwise - reuse them")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")

    args = parser.parse_args()

    output_exists = os.path.exists(args.output)

    if output_exists and not args.overwrite:
        results_df = pandas.read_csv(args.output)
    else:
        # Import the engine module
        import importlib.util

        spec = importlib.util.spec_from_file_location("engine", args.engine)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        print(f"Loading engine {args.engine} with model {args.model}...")
        transcribe_fn = engine.create_app(model_path=args.model, device=args.device)

        print(f"Loading dataset {args.dataset}...")
        dataset_parts = args.dataset.split(":")
        dataset_name = dataset_parts[0]
        dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
        ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

        if args.name:
            ds = datasets.load_dataset(dataset_name, name=args.name, trust_remote_code=True)[dataset_split]
        else:
            ds = datasets.load_dataset(dataset_name, trust_remote_code=True)[dataset_split]

        print(f"Beginning evaluation with {args.workers} workers.")
        results_df = evaluate_model(transcribe_fn, ds, ds_text_column, args.workers)

        # Add model and dataset info as columns
        results_df["model"] = args.model
        results_df["dataset"] = dataset_name
        results_df["dataset_split"] = dataset_split
        results_df["engine"] = args.engine

        results_df.to_csv(args.output, encoding="utf-8", index=False)
        print(f"Results saved to {args.output}")

    # Calculate final metrics
    metrics = calculate_final_metrics(results_df)

    print(f"Evaluation done. WER={metrics.wer}, WIL={metrics.wil}.")
