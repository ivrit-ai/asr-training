#!/usr/bin/env python3
"""Restore spaces between Whisper timestamp tokens and segment text.

Usage:
    python inject_space_to_dataset.py /path/to/result

The repaired dataset is written to ``/path/to/result-spaces-injected`` by
default. The source dataset is never modified.
"""

import argparse
import re
from pathlib import Path

TIMESTAMP_BEFORE_TEXT = re.compile(r"(<\|\d+(?:\.\d+)?\|>)(?=[^<\s])")


def restore_timestamp_spaces(transcripts: list[str]) -> dict[str, list[str]]:
    """Rewrite the selected ``transcript`` column passed by ``Dataset.map``.

    ``input_columns="transcript"`` makes a batched map call pass this function
    a list of transcript strings rather than a dictionary of dataset columns.
    ``Dataset.map`` still requires replacement values to be returned in a
    dictionary keyed by the updated column name.
    """
    return {
        "transcript": [TIMESTAMP_BEFORE_TEXT.sub(r"\1 ", transcript) for transcript in transcripts]
    }


def main() -> None:
    from datasets import load_from_disk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to a dataset previously saved with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Destination for the repaired dataset. Defaults to <dataset_path>-spaces-injected.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Optional number of worker processes for transcript rewriting.",
    )
    args = parser.parse_args()

    input_path = args.dataset_path.expanduser()
    output_path = (args.output_path or input_path.with_name(f"{input_path.name}-spaces-injected")).expanduser()

    if not input_path.is_dir():
        parser.error(f"Dataset path does not exist or is not a directory: {input_path}")
    if output_path.exists():
        parser.error(f"Output path already exists: {output_path}")

    dataset = load_from_disk(str(input_path))
    repaired_dataset = dataset.map(
        restore_timestamp_spaces,
        input_columns="transcript",
        batched=True,
        num_proc=args.num_proc,
        desc="Restoring timestamp spaces",
    )
    repaired_dataset.save_to_disk(str(output_path))
    print(f"Saved repaired dataset to {output_path}")


if __name__ == "__main__":
    main()
