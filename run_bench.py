#!/usr/bin/env python3

import argparse
import json
import os
import platform
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


def collect_machine_info() -> dict:
    """Collect CPU and GPU information about the current machine."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    
    # Try to get CPU count
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        pass
    
    # Try to get more detailed CPU info
    try:
        import psutil
        info["cpu_freq_mhz"] = psutil.cpu_freq().max if psutil.cpu_freq() else None
        info["total_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    
    # Try to get full CPU info from /proc/cpuinfo
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo_raw = f.read()
            info["cpuinfo"] = cpuinfo_raw
    except Exception:
        pass
    
    # Try to get GPU info using torch
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                }
                info["gpus"].append(gpu_info)
        else:
            info["cuda_available"] = False
    except Exception:
        info["cuda_available"] = False
    
    return info


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to engine script (e.g. engines/faster_whisper_engine.py)")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation results")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if exists")
    parser.add_argument("--device", type=str, default="auto", help="copmute device")
    parser.add_argument("--parallel-mode", type=str, default="thread", help="parallism approach for mulit-worker evaluation")
    parser.add_argument("--devices", type=str, default="", help="Devices list to use for process multi-worker evaluation (GPU Bound)")
    parser.add_argument("--batch-size", type=str, default="1", help="per transcribe call batch size")
    parser.add_argument("--benchmark-timing", type=int, default=None, metavar="N",
                        help="Run each transcription N times and store all timing results")
    args = parser.parse_args()

    # Ensure engine script exists
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine script not found: {args.engine}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect and save machine info
    machine_info = collect_machine_info()
    machine_info_path = os.path.join(args.output_dir, "machine.json")
    with open(machine_info_path, "w") as f:
        json.dump(machine_info, f, indent=2)
    print(f"Machine info saved to {machine_info_path}")

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
            args.device,
            "--batch-size",
            args.batch_size,
            "--parallel-mode",
            args.parallel_mode,
            "--devices",
            args.devices
        ]

        if args.overwrite:
            cmd.append("--overwrite")
        if ds_name:
            cmd.extend(["--name", ds_name])
        if args.benchmark_timing:
            cmd.extend(["--benchmark-timing", str(args.benchmark_timing)])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {ds_path}: {e}")
            continue


if __name__ == "__main__":
    main()
