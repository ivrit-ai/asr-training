#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import threading
import time
from queue import Queue, Empty
from typing import Iterator, Dict, Any, Optional

import datasets
import jiwer
import pandas
import whisper.normalizers
from hebrew import Hebrew
from tqdm import tqdm


def clean_some_unicode_from_text(text):
    chars_to_remove = "\u061c"  # Arabic letter mark
    chars_to_remove += "\u200b\u200c\u200d"  # Zero-width space, non-joiner, joiner
    chars_to_remove += "\u200e\u200f"  # LTR and RTL marks
    chars_to_remove += "\u202a\u202b\u202c\u202d\u202e"  # LTR/RTL embedding, pop, override
    chars_to_remove += "\u2066\u2067\u2068\u2069"  # Isolate controls
    chars_to_remove += "\ufeff"  # Zero-width no-break space
    return text.translate({ord(c): None for c in chars_to_remove})


def remove_niqqud(text: str):
    """Remove niqqud from Hebrew text."""
    return Hebrew(text).no_niqqud().string


from stable_whisper.audio import AudioLoader
import numpy as np
from create_dataset import _load_data_manifest
from stable_whisper.result import WhisperResult, Segment

from pathlib import Path
import json

WHISPER_EXPECTED_SAMPLE_RATE = 16000


def get_slice_audio_data(audio_loader: AudioLoader, slice):
    audio_start_sec = slice["seek"]
    seek_sample = int(audio_start_sec * WHISPER_EXPECTED_SAMPLE_RATE)
    slice_length_samples = int(slice["duration"] * WHISPER_EXPECTED_SAMPLE_RATE)
    audio_data = audio_loader.next_chunk(seek_sample, slice_length_samples)
    return audio_data


def get_segment_word_scores(segment: Segment) -> list[float]:
    """
    Get the word scores for a segment.
    This is a helper function to extract the word scores from a segment.
    """
    if not segment.has_words:
        return []

    # Extract word scores from the segment
    word_scores = []
    for word in segment.words:
        if hasattr(word, "probability"):
            word_scores.append(word.probability)
    return word_scores


def calculate_median_quality_score(scores: list[float]) -> float:
    """
    Calculate the median quality score for a list of scores.
    This is a helper function to calculate the median quality score for a list of scores.
    """
    # Calculate the median probability of all words in the segment
    quality_score = float(np.median(scores)) if scores else 0.0
    return quality_score


def calculate_segments_quality_score(segments: list[Segment]) -> float:
    if not segments:
        return 0.0

    """Calculate the quality score based on the median word probabilities for a single segment."""
    try:
        all_word_probs = []
        for segment in segments:
            all_word_probs.extend(get_segment_word_scores(segment))
        # Calculate the median probability of all words in the segment
        quality_score = calculate_median_quality_score(all_word_probs)
        return quality_score

    except Exception:
        return 0.0


def stream_normalized_dataset(dataset_name: str, max_entries: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    input_folder = Path(dataset_name)
    segments_filename_glob: str = "transcript.*.json"
    audio_filename_glob: str = "audio.*"
    metadata_glob: str = "metadata.json"
    per_sample_quality_threshold = 0
    per_segment_quality_threshold = 0.6
    target_slice_duration = 20
    min_segment_gap_to_slice_at = 0.5

    manifest = _load_data_manifest(
        input_folder,
        segments_glob=f"**/{segments_filename_glob}",
        audio_filename_glob=audio_filename_glob,
        metadata_glob=metadata_glob,
    )
    stream_id = 0
    for audio_file, segments_data_file, metadata_file in manifest:
        segments_data = WhisperResult(str(segments_data_file))
        segments = segments_data.segments
        sample_quality_score = None
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            sample_quality_score = metadata.get("quality_score", None)
        if (
            sample_quality_score is not None
            and per_sample_quality_threshold > 0
            and sample_quality_score < per_sample_quality_threshold
        ):
            continue

        # Load Audio (streams output from an FFMPEG process for memory efficiency)
        audio_loader = AudioLoader(
            str(audio_file),
            stream=True,
            sr=WHISPER_EXPECTED_SAMPLE_RATE,
            buffer_size=int(2 * target_slice_duration * WHISPER_EXPECTED_SAMPLE_RATE),
        )

        try:
            audio_duration = audio_loader.get_duration()

            start_at = 0
            curr_segment_idx = 0
            slices = []
            segments_in_slice = []
            while curr_segment_idx < len(segments):
                curr_segment = segments[curr_segment_idx]
                segments_in_slice.append(curr_segment)
                curr_segment_end = curr_segment.end
                curr_slice_duration = curr_segment_end - start_at
                if curr_slice_duration > target_slice_duration:
                    next_segment_idx = curr_segment_idx + 1
                    if next_segment_idx < len(segments):
                        next_segment = segments[next_segment_idx]
                        next_segment_start = next_segment.start
                        gap_to_next_segment = next_segment_start - curr_segment_end
                        if gap_to_next_segment >= min_segment_gap_to_slice_at:
                            end_at = next_segment_start - gap_to_next_segment / 2
                            slices.append(
                                {"seek": start_at, "segments": segments_in_slice, "duration": end_at - start_at}
                            )
                            segments_in_slice = []
                            start_at = end_at

                curr_segment_idx += 1

            # close final slice
            if segments_in_slice:
                slices.append({"seek": start_at, "segments": segments_in_slice, "duration": audio_duration - start_at})

            for slice in slices:
                slice_quality_score = calculate_segments_quality_score(slice["segments"])
                if slice_quality_score < per_segment_quality_threshold:
                    continue

                audio_for_slice = get_slice_audio_data(audio_loader, slice)
                text_for_slice = "".join([segment.text for segment in slice["segments"]])
                entry = {
                    "source_entry_id": metadata.get("source_entry_id", "unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "_stream_id": stream_id,
                    "audio": {
                        "array": audio_for_slice,
                        "sampling_rate": WHISPER_EXPECTED_SAMPLE_RATE,
                    },
                    "text": text_for_slice,
                }
                yield entry
                stream_id += 1

                if not (max_entries is None or stream_id < max_entries):
                    break
        finally:
            audio_loader.terminate()

        if not (max_entries is None or stream_id < max_entries):
            break


def create_streaming_dataset(
    dataset_name: str, dataset_split: str = "test", name: Optional[str] = None, max_entries: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Mock streaming dataset function that yields entries one by one.
    In a real implementation, this would stream from a remote source or large file.
    """
    print(f"Creating streaming dataset from {dataset_name}:{dataset_split}...")

    # Load the actual dataset (in real streaming, this would be different)
    if name:
        ds = datasets.load_dataset(dataset_name, name=name, trust_remote_code=True)[dataset_split]
    else:
        ds = datasets.load_dataset(dataset_name, trust_remote_code=True)[dataset_split]

    # Simulate streaming by yielding entries with a small delay
    total_entries = min(len(ds), max_entries) if max_entries else len(ds)

    for i in range(total_entries):
        entry = dict(ds[i])
        entry["_stream_id"] = i  # Add stream ID for tracking
        yield entry


class ThreadSafeCSVWriter:
    """
    Thread-safe CSV writer that appends results as they come in.
    """

    def __init__(self, output_path: str, overwrite: bool = False):
        self.output_path = output_path
        self.lock = threading.Lock()
        self.header_written = False

        if overwrite and os.path.exists(output_path):
            os.remove(output_path)

    def write_entry(self, entry_data: Dict[str, Any]):
        """Write a single entry to the CSV file in a thread-safe manner."""
        with self.lock:
            df = pandas.DataFrame([entry_data])

            # Write header if this is the first entry
            if not self.header_written:
                df.to_csv(self.output_path, mode="w", index=False, encoding="utf-8")
                self.header_written = True
            else:
                df.to_csv(self.output_path, mode="a", header=False, index=False, encoding="utf-8")


class HebrewTextNormalizer:
    def __init__(self):
        self.whisper_normalizer = whisper.normalizers.BasicTextNormalizer()

    def __call__(self, text):
        text = clean_some_unicode_from_text(text)
        text = remove_niqqud(text)
        text = text.replace('"', "").replace("'", "")

        return self.whisper_normalizer(text)


def process_entry_streaming(args):
    """Process a single entry and return the results."""
    entry, transcribe_fn, text_column, normalizer, csv_writer, model_info, skip_wer_calculation, skip_normalization = (
        args
    )

    stream_id = entry.get("_stream_id", 0)

    try:
        raw_ref_text = entry[text_column]
        raw_eval_text = transcribe_fn(entry)

        if not skip_normalization:
            ref_text = normalizer(raw_ref_text)
            eval_text = normalizer(raw_eval_text)
        else:
            ref_text = raw_ref_text
            eval_text = raw_eval_text

        if not skip_wer_calculation:
            entry_metrics = jiwer.process_words([ref_text], [eval_text])
        else:
            entry_metrics = None

        entry_data = {
            "id": stream_id,
            "reference_text": raw_ref_text,
            "predicted_text": raw_eval_text,
        }

        if not skip_normalization:
            entry_data["norm_reference_text"] = ref_text
            entry_data["norm_predicted_text"] = eval_text
        if entry_metrics:
            entry_data["wer"] = entry_metrics.wer
            entry_data["wil"] = entry_metrics.wil
            entry_data["substitutions"] = entry_metrics.substitutions
            entry_data["deletions"] = entry_metrics.deletions
            entry_data["insertions"] = entry_metrics.insertions
            entry_data["hits"] = entry_metrics.hits

        # Add model and dataset info
        entry_data.update(model_info)

        # Add metadata from entry
        for key in entry.keys():
            if key not in ["audio", text_column, "_stream_id"]:
                entry_data[f"metadata_{key}"] = entry[key]

        # Write to CSV immediately
        csv_writer.write_entry(entry_data)

        return entry_data

    except Exception as e:
        print(f"Error processing entry {stream_id}: {e}")
        return None


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


def evaluate_model_streaming(
    transcribe_fn,
    dataset_stream: Iterator[Dict[str, Any]],
    text_column: str,
    csv_writer: ThreadSafeCSVWriter,
    model_info: Dict[str, str],
    num_workers: int = 1,
    max_entries: Optional[int] = None,
    skip_wer_calculation: bool = False,
    skip_normalization: bool = False,
):
    """
    Evaluate model using streaming dataset with parallel processing.
    """
    normalizer = HebrewTextNormalizer()

    # Queue for streaming entries
    entry_queue = Queue(maxsize=num_workers * 2)  # Buffer size
    processed_count = 0
    total_entries = max_entries if max_entries else "unknown"

    # Producer thread to feed the queue from stream
    def producer():
        nonlocal processed_count
        try:
            for entry in dataset_stream:
                if max_entries and processed_count >= max_entries:
                    break
                entry_queue.put(entry)
                processed_count += 1
        finally:
            # Signal end of stream
            for _ in range(num_workers):
                entry_queue.put(None)

    # Start producer thread
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    # Process entries in parallel
    entries_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Progress bar
        pbar = tqdm(desc="Processing entries", total=max_entries)

        # Submit initial batch of workers
        futures = []

        def submit_worker(block_until_item_queued: bool = True):
            """Submit a worker to process the next entry from queue."""
            try:
                timeout = None if block_until_item_queued else 1.0
                entry = entry_queue.get(timeout=timeout)
                if entry is None:  # End of stream signal
                    return None

                args = (
                    entry,
                    transcribe_fn,
                    text_column,
                    normalizer,
                    csv_writer,
                    model_info,
                    skip_wer_calculation,
                    skip_normalization,
                )
                future = executor.submit(process_entry_streaming, args)
                return future
            except Empty:
                return None

        # Keep submitting workers as they complete
        active_futures = []

        # Submit initial batch
        for _ in range(num_workers):
            future = submit_worker(block_until_item_queued=True)
            if future:
                active_futures.append(future)

        # Process completed futures and submit new ones
        while active_futures:
            # Wait for at least one future to complete
            done_futures = []
            for future in concurrent.futures.as_completed(active_futures, timeout=0.1):
                done_futures.append(future)
                break

            # Process completed futures
            for future in done_futures:
                active_futures.remove(future)
                try:
                    result = future.result()
                    if result:
                        entries_data.append(result)
                        last_wer = result["wer"] if "wer" in result else 0
                        last_wil = result["wil"] if "wil" in result else 0
                        pbar.set_postfix(last_wer=f"{last_wer:.5f}", last_wil=f"{last_wil:.5f}")
                        pbar.update(1)
                except Exception as e:
                    print(f"Error in worker: {e}")

                # Submit new worker if stream is not exhausted
                new_future = submit_worker()
                if new_future:
                    active_futures.append(new_future)

        pbar.close()

    # Wait for producer to finish
    producer_thread.join()

    print(f"Processed {len(entries_data)} entries")
    return pandas.DataFrame(entries_data) if entries_data else pandas.DataFrame()


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
    parser.add_argument(
        "--max-entries", type=int, default=None, help="Maximum number of entries to process from the streaming dataset"
    )
    parser.add_argument(
        "--skip-wer-calculation",
        action="store_true",
        help="Skip calculating WER and other metrics and only save the transcription",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalizing the transcription and only save the transcription",
    )

    args = parser.parse_args()

    output_exists = os.path.exists(args.output)

    if output_exists and not args.overwrite:
        print(f"Loading existing results from {args.output}")
        results_df = pandas.read_csv(args.output)
    else:
        # Import the engine module
        import importlib.util

        spec = importlib.util.spec_from_file_location("engine", args.engine)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        print(f"Creating streaming dataset from {args.dataset}...")
        dataset_parts = args.dataset.split(":")
        dataset_name = dataset_parts[0]
        dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
        ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

        print(f"Loading engine {args.engine} with model {args.model}...")
        transcribe_fn = engine.create_app(model_path=args.model, device=args.device, text_column=ds_text_column)

        # Create streaming dataset
        # dataset_stream = create_streaming_dataset(
        #     dataset_name=dataset_name, dataset_split=dataset_split, name=args.name, max_entries=args.max_entries
        # )
        dataset_stream = stream_normalized_dataset(dataset_name=dataset_name, max_entries=args.max_entries)

        # Create thread-safe CSV writer
        csv_writer = ThreadSafeCSVWriter(args.output, overwrite=args.overwrite)

        # Model info to add to each entry
        model_info = {
            "model": args.model,
            "dataset": dataset_name,
            "dataset_split": dataset_split,
            "engine": args.engine,
        }

        print(f"Beginning streaming evaluation with {args.workers} workers...")
        if args.max_entries:
            print(f"Processing maximum {args.max_entries} entries")

        results_df = evaluate_model_streaming(
            transcribe_fn=transcribe_fn,
            dataset_stream=dataset_stream,
            text_column=ds_text_column,
            csv_writer=csv_writer,
            model_info=model_info,
            num_workers=args.workers,
            max_entries=args.max_entries,
            skip_wer_calculation=args.skip_wer_calculation,
            skip_normalization=args.skip_normalization,
        )

        print(f"Results saved to {args.output}")

    # Calculate final metrics
    if not args.skip_wer_calculation:
        metrics = calculate_final_metrics(results_df)
        print(f"Evaluation done. WER={metrics.wer}, WIL={metrics.wil}.")
    else:
        print("Evaluation done.")
