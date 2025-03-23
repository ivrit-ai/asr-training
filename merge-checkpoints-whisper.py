import argparse
import os
import shutil

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration


def get_tensor_keys(filepath: str) -> list[str]:
    with safe_open(filepath, framework="pt", device="cpu") as f:
        return list(f.keys())


def average_checkpoints(ckps: list[str], output_model: str, weights: list[float] | None = None):
    if weights:
        assert len(weights) == len(ckps), "Number of weights must match number of checkpoints."
        weight_sum = sum(weights)
        assert abs(weight_sum - 1.0) < 1e-6, "Weights must sum to 1."
    else:
        weights = [1.0 / len(ckps)] * len(ckps)

    weight_files = sorted([f for f in sorted(os.listdir(ckp)) if f.endswith(".safetensors")] for ckp in ckps)

    for wf1, wf2 in [(wf1, wf2) for wf1 in weight_files for wf2 in weight_files]:
        assert wf1 == wf2, "Model checkpoints have different safetensor files!"

    os.makedirs(output_model, exist_ok=True)

    for weight_file in weight_files[0]:
        keys_structure = None
        averaged_state_dict = {}
        metadata = None

        for ckp_idx, ckp in enumerate(ckps):
            file = os.path.join(ckp, weight_file)
            weight = weights[ckp_idx]
            is_last_ckp = ckp_idx == len(ckps) - 1

            if metadata is None:
                with safe_open(file, framework="pt", device="cpu") as f:
                    metadata = f.metadata()

            keys = get_tensor_keys(file)
            if keys_structure is None:
                keys_structure = keys
            else:
                assert keys == keys_structure, f"Mismatch in keys for {ckp}/{weight_file} compared to first checkpoint."

            keys_chunk_size = 20
            keys_chunked = [keys[i : i + keys_chunk_size] for i in range(0, len(keys), keys_chunk_size)]

            for chunk in tqdm(keys_chunked, desc=f"Processing chunks in {weight_file}"):
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in tqdm(chunk, desc=f"{ckp}", leave=False):
                        tensor = f.get_tensor(key).to(torch.float64)
                        if key not in averaged_state_dict:
                            averaged_state_dict[key] = tensor * weight
                        else:
                            averaged_state_dict[key] += tensor * weight

                        if is_last_ckp:
                            averaged_state_dict[key] = averaged_state_dict[key].to(torch.float32)

        save_file(
            averaged_state_dict,
            os.path.join(output_model, weight_file),
            metadata=metadata,
        )

    model = WhisperForConditionalGeneration.from_pretrained(ckps[0])
    model.config.save_pretrained(output_model)

    files_to_copy = [
        "tokenizer_config.json",
        "generation_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "merges.txt",
    ]
    for file in files_to_copy:
        src = os.path.join(ckps[0], file)
        if os.path.exists(src):
            shutil.copy(src, output_model)

    print(f"Averaged model saved to: {output_model}")


def main():
    parser = argparse.ArgumentParser(
        description="Average safetensor checkpoints from multiple Whisper model directories."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint directories (must contain .safetensors files)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of weights (must sum to 1.0 and match number of checkpoints)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory for the averaged model",
    )

    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.output, args.weights)


if __name__ == "__main__":
    main()