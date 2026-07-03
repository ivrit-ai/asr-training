#!/usr/bin/env python3
# coding: utf-8
"""
Prepare a Whisper model + processor for training with an OPTIMIZED (Hebrew) tokenizer.

Given:
  - a base Whisper model (the one you will fine-tune, e.g. whisper-large-v3-turbo)
  - the optimized `step3_hebrew_tokenizer.json` produced by the tokenizer pipeline

this script produces a single self-contained directory containing:
  - the custom WhisperTokenizerFast (built from step3_hebrew_tokenizer.json)
  - the base model's feature extractor (so it forms a valid WhisperProcessor)
  - the model with its token embeddings RESIZED to the new vocab and WARM-STARTED
    (embedding salvage: exact-match reuse + subword-mean for new tokens)
  - config / generation_config token ids rewritten for the new vocab layout

Point `train-whisper.py --model_name <this_dir>` at the result — both the processor
and the model load from it unchanged.

Why this works: Step 3 preserved Whisper's special tokens in their ORIGINAL order and
appended them contiguously after the BPE vocab. So the invariants Whisper relies on
still hold: `<|notimestamps|> + 1 == <|0.00|>` and the last id is the last timestamp.

Example:
  python prepare_optimized_model.py \
    --base_model openai/whisper-large-v3-turbo \
    --tokenizer_json /path/to/outputs/step3_hebrew_tokenizer.json \
    --output_dir ./whisper-turbo-he-optimized-init
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
from tokenizers import Tokenizer
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES


# --------------------------------------------------------------------------------------
# Byte-level helpers (GPT-2 / Whisper ByteLevel) for the subword-mean warm start.
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _byte_decoder() -> Dict[str, int]:
    return {c: b for b, c in _bytes_to_unicode().items()}


def _token_to_surface(token: str, byte_decoder: Dict[str, int]) -> str:
    """Byte-level token string ('Ġthe') -> surface text (' the'). '' if not decodable."""
    try:
        raw = bytes(byte_decoder[c] for c in token)
    except KeyError:
        return ""
    return raw.decode("utf-8", errors="replace")


def build_tokenizer(tokenizer_json: str) -> WhisperTokenizerFast:
    """Construct a fully-functional WhisperTokenizerFast from the optimized tokenizer.json."""
    return WhisperTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="<|endoftext|>",
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )


def salvage_embeddings(
    old_backend: Tokenizer,
    new_backend: Tokenizer,
    old_emb: torch.Tensor,
    d_model: int,
) -> (torch.Tensor, Dict):
    """
    Build the new embedding matrix, reusing old embeddings where possible:
      1) exact string match (reused subwords + all special tokens) -> copy
      2) new token -> decode to surface, re-tokenize with old, average old embeddings
      3) unrecoverable -> small random init (should be ~0)
    """
    old_vocab = old_backend.get_vocab()
    new_vocab = new_backend.get_vocab()
    new_id_to_token = {i: t for t, i in new_vocab.items()}
    byte_decoder = _byte_decoder()

    new_emb = torch.empty(len(new_vocab), d_model, dtype=old_emb.dtype)
    torch.nn.init.normal_(new_emb, mean=0.0, std=0.02)

    stats = {"exact": 0, "special_exact": 0, "subword_mean": 0, "random_init": 0}

    for new_id, token in new_id_to_token.items():
        old_id = old_vocab.get(token)
        if old_id is not None:
            new_emb[new_id] = old_emb[old_id]
            if token.startswith("<|") and token.endswith("|>"):
                stats["special_exact"] += 1
            else:
                stats["exact"] += 1
            continue

        surface = _token_to_surface(token, byte_decoder)
        if surface:
            old_ids = old_backend.encode(surface, add_special_tokens=False).ids
            if old_ids:
                new_emb[new_id] = old_emb[old_ids].mean(dim=0)
                stats["subword_mean"] += 1
                continue

        stats["random_init"] += 1

    reused = stats["exact"] + stats["special_exact"] + stats["subword_mean"]
    stats["coverage_percent"] = round(reused / len(new_vocab) * 100, 2)
    stats["new_vocab_size"] = len(new_vocab)
    return new_emb, stats


def rewrite_token_ids(model, tokenizer: WhisperTokenizerFast) -> Dict:
    """Rewrite config + generation_config token ids to match the new vocab layout."""

    def sid(t: str) -> int:
        return tokenizer.convert_tokens_to_ids(t)

    sot = sid("<|startoftranscript|>")
    eot = sid("<|endoftext|>")
    notimestamps = sid("<|notimestamps|>")

    # Model config
    model.config.vocab_size = len(tokenizer)
    model.config.decoder_start_token_id = sot
    model.config.eos_token_id = eot
    model.config.pad_token_id = eot
    model.config.bos_token_id = eot
    model.config.max_length = model.config.max_length  # unchanged
    # Training clears these anyway; set safe defaults.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.begin_suppress_tokens = [eot]

    # Generation config (used by eval's predict_with_generate + model.generate(language=,task=))
    gc = model.generation_config
    gc.decoder_start_token_id = sot
    gc.eos_token_id = eot
    gc.pad_token_id = eot
    gc.bos_token_id = eot
    gc.no_timestamps_token_id = notimestamps
    gc.forced_decoder_ids = None
    gc.suppress_tokens = []
    gc.begin_suppress_tokens = [eot]
    # lang/task id maps rebuilt for the new layout
    gc.lang_to_id = {
        f"<|{code}|>": sid(f"<|{code}|>")
        for code in LANGUAGES
        if f"<|{code}|>" in tokenizer.get_vocab()
    }
    gc.task_to_id = {
        "transcribe": sid("<|transcribe|>"),
        "translate": sid("<|translate|>"),
    }

    return {
        "sot": sot,
        "eot": eot,
        "notimestamps": notimestamps,
        "ts_begin": notimestamps + 1,
        "last_id": len(tokenizer) - 1,
        "num_languages": len(gc.lang_to_id),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="openai/whisper-large-v3-turbo",
                   help="Model to fine-tune (its architecture / embeddings are the salvage source)")
    p.add_argument("--tokenizer_json", required=True,
                   help="Path to step3_hebrew_tokenizer.json from the tokenizer pipeline")
    p.add_argument("--embedding_source_model", default=None,
                   help="Model to salvage embeddings FROM (default: --base_model). Must share d_model.")
    p.add_argument("--base_tokenizer", default=None,
                   help="Tokenizer matching the embedding source (default: --base_model)")
    p.add_argument("--output_dir", required=True, help="Output directory for the prepared model+processor")
    return p.parse_args()


def main():
    args = parse_args()
    source_model = args.embedding_source_model or args.base_model
    base_tokenizer = args.base_tokenizer or args.base_model
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARE OPTIMIZED WHISPER MODEL + PROCESSOR")
    print("=" * 80)

    # 1) Custom tokenizer + feature extractor, saved side-by-side so that
    #    WhisperProcessor.from_pretrained(out) reassembles them (as training does).
    #    (We don't construct WhisperProcessor() directly: some transformers versions
    #    reject a manually-built fast tokenizer; the from_pretrained path is fine.)
    print(f"\nBuilding custom tokenizer from {args.tokenizer_json}")
    tokenizer = build_tokenizer(args.tokenizer_json)
    tokenizer.save_pretrained(str(out))
    WhisperFeatureExtractor.from_pretrained(args.base_model).save_pretrained(str(out))
    print(f"  New vocab size: {len(tokenizer):,}")

    # 2) Load the model we will fine-tune
    print(f"\nLoading model {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    d_model = model.config.d_model

    # 3) Salvage embeddings from the source model
    print(f"Loading embedding source: {source_model}")
    if source_model == args.base_model:
        src_model = model
    else:
        src_model = WhisperForConditionalGeneration.from_pretrained(source_model)
        assert src_model.config.d_model == d_model, (
            f"Embedding source d_model {src_model.config.d_model} != base {d_model}. "
            "Salvage source must share the model dimension."
        )

    old_backend = WhisperTokenizerFast.from_pretrained(base_tokenizer).backend_tokenizer
    new_backend = tokenizer.backend_tokenizer
    old_emb = src_model.model.decoder.embed_tokens.weight.data

    print("Salvaging embeddings (warm start)...")
    new_emb, stats = salvage_embeddings(old_backend, new_backend, old_emb, d_model)
    print(f"  exact(subword):   {stats['exact']:,}")
    print(f"  exact(special):   {stats['special_exact']:,}")
    print(f"  subword-mean:     {stats['subword_mean']:,}")
    print(f"  random-init:      {stats['random_init']:,}")
    print(f"  warm-start coverage: {stats['coverage_percent']}%")

    # 4) Resize + install the salvaged embeddings (proj_out is tied to embed_tokens)
    print("\nResizing token embeddings and installing salvaged weights...")
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.model.decoder.embed_tokens.weight.data.copy_(new_emb)
    model.tie_weights()

    # 5) Rewrite config / generation_config token ids
    ids = rewrite_token_ids(model, tokenizer)
    print(f"  token ids: sot={ids['sot']} eot={ids['eot']} notimestamps={ids['notimestamps']} "
          f"ts_begin={ids['ts_begin']} last_id={ids['last_id']} langs={ids['num_languages']}")

    # 6) Save the prepared model into the same dir as the processor
    print(f"\nSaving prepared model + processor to {out}")
    model.save_pretrained(str(out))

    # 7) Sanity checks
    assert ids["ts_begin"] == tokenizer.convert_tokens_to_ids("<|0.00|>"), "timestamp begin mismatch"
    assert ids["last_id"] == tokenizer.convert_tokens_to_ids("<|30.00|>"), "last timestamp mismatch"
    emb_rows = model.model.decoder.embed_tokens.weight.shape[0]
    assert emb_rows == len(tokenizer), f"embedding rows {emb_rows} != vocab {len(tokenizer)}"
    # The dir must load as a valid WhisperProcessor (this is how training loads it).
    proc = WhisperProcessor.from_pretrained(str(out), language="hebrew", task="transcribe")
    assert proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>") == ids["sot"]
    print("  processor reloads OK; embeddings match vocab; timestamp layout consistent")

    report = {"base_model": args.base_model, "embedding_source": source_model,
              "output_dir": str(out), "vocab_size": len(tokenizer),
              "token_ids": ids, "salvage": stats}
    (out / "optimization_init_report.json").write_text(json.dumps(report, indent=2))

    print("\n✅ Done. Train with:")
    print(f"     python train-whisper.py --model_name {out} ... [--freeze_encoder] [--embedding_warmup_steps N]")


if __name__ == "__main__":
    main()
