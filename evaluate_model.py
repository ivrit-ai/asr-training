#!/usr/bin/env python3

import argparse
import concurrent.futures
import importlib.util
import json
import os

import datasets
import jiwer
import pandas
import torch.multiprocessing as mp
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
    batch_start_i, entries, transcribe_fn, text_column, normalizer, benchmark_timing = args

    if not isinstance(entries, list):
        entries = [entries]

    results = transcribe_fn(entries)
    if not isinstance(results, list):
        results = [results]

    entry_data_list = []
    for j, (entry, (raw_eval_text, transcription_time)) in enumerate(zip(entries, results)):
        i = batch_start_i + j
        
        if isinstance(entry, str):
            raw_ref_text = ""
            entry = {"audio": entry}
        else:
            raw_ref_text = entry[text_column]
        
        # If benchmark_timing is set, run additional iterations and collect all times
        if benchmark_timing and benchmark_timing > 1:
            all_times = [transcription_time]
            for _ in range(benchmark_timing - 1):
                res = transcribe_fn([entry])  # call single for benchmark
                if isinstance(res, list):
                    res = res[0]
                _, t = res
                all_times.append(t)
            transcription_times_json = json.dumps(all_times)
        else:
            transcription_times_json = None

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
            "audio_duration": len(entry["audio"]["array"]) / entry["audio"]["sampling_rate"] if isinstance(entry["audio"], dict) else 0.0,
            "transcription_time": transcription_time,
        }
        
        if transcription_times_json:
            entry_data["transcription_times"] = transcription_times_json

        for key in entry.keys():
            if key not in ["audio", text_column]:
                entry_data[f"metadata_{key}"] = entry[key]

        entry_data_list.append(entry_data)

    return entry_data_list


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


def calculate_transcription_time_stats(df: pandas.DataFrame):
    """Calculate transcription time statistics for segments >= 5 seconds: per audio second, per output character, and raw transcription time"""
    
    # Calculate audio durations and text lengths, filtering segments >= 5 seconds
    audio_durations = []
    text_lengths = []
    transcription_times = []
    
    for _, row in df.iterrows():
        # Get audio duration from stored value
        audio_duration = row.get("audio_duration", 1.0)  # Default to 1 second if not available
        
        # Skip segments shorter than 5 seconds
        if audio_duration < 5.0:
            continue
            
        audio_durations.append(audio_duration)
        
        # Get text length (number of characters)
        text_length = len(row["predicted_text"]) if row["predicted_text"] else 0
        text_lengths.append(text_length)
        
        # Get transcription time
        transcription_time = row.get("transcription_time", 0)
        transcription_times.append(transcription_time)
    
    # Calculate normalized times
    time_per_second = [t / d if d > 0 else 0 for t, d in zip(transcription_times, audio_durations)]
    time_per_char = [t / l if l > 0 else 0 for t, l in zip(transcription_times, text_lengths)]
    
    # Calculate statistics
    def calculate_percentiles(data):
        data = [x for x in data if x > 0]  # Filter out zeros
        if not data:
            return {"mean": 0, "median": 0, "p90": 0, "p99": 0}
        
        data.sort()
        n = len(data)
        return {
            "mean": sum(data) / n,
            "median": data[n // 2] if n % 2 == 1 else (data[n // 2 - 1] + data[n // 2]) / 2,
            "p90": data[int(0.9 * n)] if n > 0 else 0,
            "p99": data[int(0.99 * n)] if n > 0 else 0
        }
    
    time_per_second_stats = calculate_percentiles(time_per_second)
    time_per_char_stats = calculate_percentiles(time_per_char)
    raw_time_stats = calculate_percentiles(transcription_times)
    
    return {
        "time_per_second": time_per_second_stats,
        "time_per_char": time_per_char_stats,
        "raw_time": raw_time_stats
    }

def _mp_worker(
    rank: int,
    device: str,
    engine_path: str,
    engine_kwargs: dict,
    dataset_name: str,
    dataset_name_arg,
    dataset_split: str,
    shard_indices: list,
    text_column: str,
    benchmark_timing,
    batch_size: int,
    result_queue,
):
    """Worker process: loads the engine and model on the assigned device, processes its dataset shard.

    Each spawned process calls this function independently. The engine module is
    re-imported here (inside the child process) so that no CUDA context or loaded
    model is inherited from the parent — a requirement when using the 'spawn'
    start method.
    """
    try:
        # Load engine module fresh inside this process
        spec = importlib.util.spec_from_file_location("engine", engine_path)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        # Override device with the one assigned to this worker
        engine_kwargs = dict(engine_kwargs)
        engine_kwargs["device"] = device
        print(f"[Worker {rank}] Loading engine {engine_path!r} on device {device!r}...")
        transcribe_fn = engine.create_app(**engine_kwargs)

        # Each worker loads the dataset independently to avoid serializing
        # large audio arrays across process boundaries.
        print(f"[Worker {rank}] Loading dataset {dataset_name!r} ({dataset_split})...")
        if dataset_name_arg:
            ds = datasets.load_dataset(dataset_name, name=dataset_name_arg, trust_remote_code=True)[dataset_split]
        else:
            ds = datasets.load_dataset(dataset_name, trust_remote_code=True)[dataset_split]

        normalizer = HebrewTextNormalizer()

        if batch_size == 1:
            for idx in shard_indices:
                entry = ds[idx]
                batch_results = process_entry((idx, entry, transcribe_fn, text_column, normalizer, benchmark_timing))
                result_queue.put(batch_results)
        else:
            for i in range(0, len(shard_indices), batch_size):
                batch_idx = shard_indices[i : i + batch_size]
                batch = list(ds.select(batch_idx))
                batch_results = process_entry((batch_idx[0], batch, transcribe_fn, text_column, normalizer, benchmark_timing))
                result_queue.put(batch_results)

    except Exception as exc:
        # Surface the exception to the main process via the queue so we don't hang
        result_queue.put(exc)

    finally:
        # Sentinel: signals to the main process that this worker is done
        result_queue.put(None)


def evaluate_model_multiprocess(
    engine_path: str,
    engine_kwargs: dict,
    dataset_name: str,
    dataset_name_arg,
    dataset_split: str,
    ds_length: int,
    text_column: str,
    devices: list,
    benchmark_timing=None,
    batch_size: int = 1,
) -> pandas.DataFrame:
    """Evaluate a model using one worker process per device.

    Each worker calls ``create_app`` internally so that models are loaded
    directly onto their target GPU inside the child process, which is required
    when using ``torch.multiprocessing`` with the ``'spawn'`` start method
    (CUDA contexts cannot be forked).

    Args:
        engine_path: Path to the engine Python file.
        engine_kwargs: Keyword arguments forwarded to ``engine.create_app``
            (``device`` is overridden per worker).
        dataset_name: HuggingFace dataset identifier.
        dataset_name_arg: Optional ``name`` argument for ``load_dataset``.
        dataset_split: Dataset split (e.g., ``"test"``).
        ds_length: Total number of entries in the split (used for sharding
            and progress tracking without loading the dataset in the main
            process).
        text_column: Name of the column containing reference transcriptions.
        devices: List of device strings, one per worker process
            (e.g., ``["cuda:0", "cuda:1"]``).
        benchmark_timing: If set, each entry is transcribed this many times
            and all timing values are stored.
        batch_size: Number of entries per transcription call.

    Returns:
        A :class:`pandas.DataFrame` with one row per dataset entry, sorted
        by original index.
    """
    # 'spawn' is required for CUDA: forked processes inherit the parent's CUDA
    # context which leads to crashes/corruption.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    num_workers = len(devices)

    # Round-robin shard assignment so load is spread evenly regardless of
    # whether entries vary in duration.
    shards = [[] for _ in range(num_workers)]
    for i in range(ds_length):
        shards[i % num_workers].append(i)

    result_queue = mp.Queue()

    # Spawn one process per device
    processes = []
    for rank, device in enumerate(devices):
        p = mp.Process(
            target=_mp_worker,
            args=(
                rank,
                device,
                engine_path,
                engine_kwargs,
                dataset_name,
                dataset_name_arg,
                dataset_split,
                shards[rank],
                text_column,
                benchmark_timing,
                batch_size,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results from all workers, tracking progress in real time
    entries_data = []
    finished_workers = 0
    errors = []

    with tqdm(total=ds_length, desc=f"Processing entries ({num_workers} processes)") as pbar:
        while finished_workers < num_workers:
            # Poll with a short timeout so we can detect dead workers
            try:
                result = result_queue.get(timeout=30)
            except Exception:
                # Check whether any worker died unexpectedly
                for p in processes:
                    if not p.is_alive() and p.exitcode not in (None, 0):
                        errors.append(f"Worker process (pid={p.pid}) exited with code {p.exitcode}")
                if errors:
                    break
                continue

            if result is None:
                # Sentinel — one worker finished cleanly
                finished_workers += 1
                continue

            if isinstance(result, Exception):
                errors.append(str(result))
                # Count this worker as finished (it sent None in finally block)
                continue

            entries_data.extend(result)
            if result:
                last = result[-1]
                pbar.set_postfix(last_wer=f"{last['wer']:.5f}", last_wil=f"{last['wil']:.5f}")
            pbar.update(len(result))

    # Ensure all child processes have terminated
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    if errors:
        raise RuntimeError("One or more worker processes failed:\n" + "\n".join(errors))

    entries_data.sort(key=lambda x: x["id"])
    return pandas.DataFrame(entries_data)


def evaluate_model(transcribe_fn, ds, text_column, num_workers=1, benchmark_timing=None, batch_size=1):
    normalizer = HebrewTextNormalizer()
    entries_data = []

    # Prepare arguments for parallel processing
    if batch_size == 1:
        process_args = [(i, ds[i], transcribe_fn, text_column, normalizer, benchmark_timing) for i in range(len(ds))]
    else:
        process_args = []
        for i in range(0, len(ds), batch_size):
            batch_indices = range(i, min(i + batch_size, len(ds)))
            batch = list(ds.select(batch_indices))
            process_args.append((i, batch, transcribe_fn, text_column, normalizer, benchmark_timing))

    # Process entries in parallel with progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = [executor.submit(process_entry, arg) for arg in process_args]

        # Use tqdm to track progress
        entries_data = []
        with tqdm(total=len(ds), desc="Processing entries") as pbar:
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                entries_data.extend(batch_results)
                if batch_results:
                    last_wer = batch_results[-1]["wer"]
                    last_wil = batch_results[-1]["wil"]
                    pbar.set_postfix(last_wer=f"{last_wer:.5f}", last_wil=f"{last_wil:.5f}")
                pbar.update(len(batch_results))

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
    parser.add_argument("--benchmark-timing", type=int, default=None, metavar="N",
                        help="Run each transcription N times and store all timing results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for transcription processing")
    parser.add_argument("--parallel-mode", type=str, choices=["thread", "process"], default="thread",
                        help="Parallelism backend: 'thread' (default, ThreadPoolExecutor) or "
                             "'process' (torch.multiprocessing with one process per device)")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated device list for process mode (e.g. 'cuda:0,cuda:1'). "
                             "The number of devices sets the number of worker processes. "
                             "Ignored in thread mode (use --device instead).")

    args = parser.parse_args()

    # Parse dataset info from command line argument
    dataset_parts = args.dataset.split(":")
    dataset_name = dataset_parts[0]
    dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
    ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

    output_exists = os.path.exists(args.output)

    if output_exists and not args.overwrite:
        results_df = pandas.read_csv(args.output)
    elif args.parallel_mode == "process":
        # --- Process-based parallelism via torch.multiprocessing ---
        # Each worker process calls create_app internally, so the main process
        # never loads the model. The dataset is also loaded inside each worker.
        if not args.devices:
            parser.error("--devices is required when --parallel-mode=process (e.g. --devices cuda:0,cuda:1)")

        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        if not devices:
            parser.error("--devices must contain at least one device string")

        engine_kwargs = {"model_path": args.model}
        # Note: 'device' is overridden per worker; we still forward any
        # extra engine kwargs via the dict if needed in the future.

        # Load dataset in the main process only to determine its length for
        # sharding. We avoid loading audio here to keep memory lean; each
        # worker reloads the full dataset.
        print(f"Loading dataset {args.dataset} (to determine length)...")
        if args.name:
            ds_meta = datasets.load_dataset(dataset_name, name=args.name, trust_remote_code=True)[dataset_split]
        else:
            ds_meta = datasets.load_dataset(dataset_name, trust_remote_code=True)[dataset_split]
        ds_length = len(ds_meta)
        del ds_meta  # free before spawning to keep parent memory footprint small

        print(f"Beginning evaluation with {len(devices)} worker processes on devices: {devices}")
        results_df = evaluate_model_multiprocess(
            engine_path=args.engine,
            engine_kwargs=engine_kwargs,
            dataset_name=dataset_name,
            dataset_name_arg=args.name,
            dataset_split=dataset_split,
            ds_length=ds_length,
            text_column=ds_text_column,
            devices=devices,
            benchmark_timing=args.benchmark_timing,
            batch_size=args.batch_size,
        )

        # Add model and dataset info as columns
        results_df["model"] = args.model
        results_df["dataset"] = dataset_name
        results_df["dataset_split"] = dataset_split
        results_df["engine"] = args.engine

        results_df.to_csv(args.output, encoding="utf-8", index=False)
        print(f"Results saved to {args.output}")
    else:
        # --- Thread-based parallelism (original behavior) ---
        spec = importlib.util.spec_from_file_location("engine", args.engine)
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        print(f"Loading engine {args.engine} with model {args.model}...")
        transcribe_fn = engine.create_app(model_path=args.model, device=args.device)

        print(f"Loading dataset {args.dataset}...")
        if args.name:
            ds = datasets.load_dataset(dataset_name, name=args.name, trust_remote_code=True)[dataset_split]
        else:
            ds = datasets.load_dataset(dataset_name, trust_remote_code=True)[dataset_split]

        print(f"Beginning evaluation with {args.workers} workers.")
        results_df = evaluate_model(transcribe_fn, ds, ds_text_column, args.workers, args.benchmark_timing, args.batch_size)

        # Add model and dataset info as columns
        results_df["model"] = args.model
        results_df["dataset"] = dataset_name
        results_df["dataset_split"] = dataset_split
        results_df["engine"] = args.engine

        results_df.to_csv(args.output, encoding="utf-8", index=False)
        print(f"Results saved to {args.output}")

    # Calculate final metrics
    metrics = calculate_final_metrics(results_df)
    
    # Calculate transcription time statistics
    time_stats = calculate_transcription_time_stats(results_df)

    print(f"Evaluation done. WER={metrics.wer}, WIL={metrics.wil}.")
    
    # Print one-liner results
    workload = f"{dataset_name}:{dataset_split}"
    print(f"{workload}\t{time_stats['raw_time']['mean']:.3f}\t{time_stats['raw_time']['median']:.3f}\t{time_stats['raw_time']['p90']:.3f}\t{time_stats['raw_time']['p99']:.3f}\t{time_stats['time_per_second']['mean']:.3f}\t{time_stats['time_per_second']['median']:.3f}\t{time_stats['time_per_second']['p90']:.3f}\t{time_stats['time_per_second']['p99']:.3f}\t{time_stats['time_per_char']['mean']:.3f}\t{time_stats['time_per_char']['median']:.3f}\t{time_stats['time_per_char']['p90']:.3f}\t{time_stats['time_per_char']['p99']:.3f}")
