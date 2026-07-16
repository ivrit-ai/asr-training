#!/usr/bin/env python3
# coding: utf-8
"""
Smoke-test a prepared optimized-tokenizer Whisper model BEFORE committing GPU hours.

Loads the directory produced by `prepare_optimized_model.py` and checks, on CPU:
  1. Processor + model load, and vocab sizes agree (tokenizer == config == embeddings).
  2. proj_out is tied to the decoder token embeddings.
  3. Special-token layout is consistent (notimestamps+1==<|0.00|>, last id==<|30.00|>).
  4. A forward pass on a dummy batch yields logits of width == new vocab size.
  5. Loss is finite for a teacher-forced dummy batch (labels shifted like training).
  6. generate() runs and only emits in-range token ids.
  7. A Hebrew/English string round-trips through encode/decode.

Usage:
  python verify_optimized_model.py --model_dir ./whisper-turbo-he-optimized-init
"""

import argparse

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True, help="Directory from prepare_optimized_model.py")
    p.add_argument("--target_language", default="hebrew")
    p.add_argument("--sample_text", default="שלום עולם, this is a quick test")
    return p.parse_args()


def check(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    args = parse_args()
    print("=" * 80)
    print(f"VERIFY OPTIMIZED MODEL: {args.model_dir}")
    print("=" * 80)

    processor = WhisperProcessor.from_pretrained(
        args.model_dir, language=args.target_language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()

    tok = processor.tokenizer
    vocab = len(tok)
    results = []

    # 1) Vocab sizes agree across tokenizer / config / embedding matrix
    emb_rows = model.model.decoder.embed_tokens.weight.shape[0]
    proj_rows = model.proj_out.weight.shape[0]
    print("\n[1] Vocab-size agreement")
    results.append(check("tokenizer == config.vocab_size", vocab == model.config.vocab_size,
                         f"{vocab} vs {model.config.vocab_size}"))
    results.append(check("tokenizer == embed_tokens rows", vocab == emb_rows, f"{vocab} vs {emb_rows}"))
    results.append(check("tokenizer == proj_out rows", vocab == proj_rows, f"{vocab} vs {proj_rows}"))

    # 2) proj_out tied to input embeddings
    print("\n[2] Weight tying")
    tied = model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr()
    results.append(check("proj_out tied to embed_tokens", tied))

    # 3) Special-token layout invariants
    print("\n[3] Special-token layout")
    special_tokens = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|transcribe|>",
        "<|he|>",
        "<|startofprev|>",
        "<|notimestamps|>",
        "<|0.00|>",
    ]
    print("  Special-token IDs:")
    for token in special_tokens:
        print(f"    {token} -> {tok.convert_tokens_to_ids(token)}")

    nots = tok.convert_tokens_to_ids("<|notimestamps|>")
    results.append(check("notimestamps+1 == <|0.00|>", nots + 1 == tok.convert_tokens_to_ids("<|0.00|>")))
    results.append(check("last id == <|30.00|>", vocab - 1 == tok.convert_tokens_to_ids("<|30.00|>")))
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    results.append(check("config.decoder_start_token_id == <|startoftranscript|>",
                         model.config.decoder_start_token_id == sot,
                         f"{model.config.decoder_start_token_id} vs {sot}"))

    # 4) + 5) Forward pass on a dummy batch -> logits width and finite loss
    print("\n[4/5] Forward pass (dummy batch)")
    num_mel = model.config.num_mel_bins
    input_features = torch.randn(1, num_mel, 2 * model.config.max_source_positions)
    tok.set_prefix_tokens(language=args.target_language, task="transcribe", predict_timestamps=False)
    prefix = tok.prefix_tokens
    text_ids = tok(args.sample_text, add_special_tokens=False, add_prefix_space=True)["input_ids"]
    seq = prefix + text_ids + [tok.convert_tokens_to_ids("<|endoftext|>")]
    seq = torch.tensor([seq])
    decoder_input_ids = seq[:, :-1]
    labels = seq[:, 1:].clone()

    with torch.no_grad():
        out = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        logits = out.logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)
        )
    results.append(check("logits width == new vocab", logits.shape[-1] == vocab,
                         f"{tuple(logits.shape)}"))
    results.append(check("loss is finite", torch.isfinite(loss).item(), f"loss={loss.item():.3f}"))

    # 6) generate() runs and stays in range
    print("\n[6] generate()")
    with torch.no_grad():
        gen = model.generate(
            input_features=input_features,
            language=args.target_language,
            task="transcribe",
            max_new_tokens=8,
        )
    results.append(check("generated ids all < vocab", int(gen.max()) < vocab,
                         f"max id {int(gen.max())} < {vocab}"))

    # 7) Round-trip
    print("\n[7] Text round-trip")
    ids = tok(args.sample_text, add_special_tokens=False, add_prefix_space=True)["input_ids"]
    decoded = tok.decode(ids)
    results.append(check("encode/decode round-trip", decoded.strip() == args.sample_text.strip(),
                         f"'{decoded}'"))

    print("\n" + "=" * 80)
    if all(results):
        print("✅ ALL CHECKS PASSED — model is safe to train.")
        return 0
    print(f"❌ {results.count(False)} CHECK(S) FAILED — do not train until resolved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
