"""End-to-end sanity tests for prepare_training_dataset.

These build a tiny synthetic input tree (real WAV audio via ffmpeg + matching
transcript/metadata JSON) in the exact on-disk layout the pipeline expects, then
run the *real* prepare_training_dataset. They are intentionally small (a few
short synthetic files) so they are safe to run locally and in CI without touching
real 10k-hour data.

Goal: prove the pipeline is stable and correct (schema, row counts, audio
decodable, sampling rate stamped) BEFORE/AFTER the concat/cast performance fix.
"""

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from create_dataset import prepare_training_dataset

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg CLI required for audio decoding"
)

SR = 16000


def _write_sine_wav(path: Path, duration_sec: float, freq: float = 220.0) -> None:
    """Write a mono 16kHz PCM16 sine-wave WAV of the given duration."""
    n = int(duration_sec * SR)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for i in range(n):
            val = int(0.3 * 32767 * math.sin(2 * math.pi * freq * (i / SR)))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))


def _make_transcript(duration_sec: float, seg_len: float = 2.0) -> dict:
    """Build a WhisperResult-compatible dict with word-level probabilities.

    One segment every `seg_len` seconds so that a 30s slice contains several
    segments and multi-slice files exercise the prev_transcript path.
    """
    segments = []
    t = 0.0
    while t + seg_len <= duration_sec:
        segments.append(
            {
                "start": round(t, 2),
                "end": round(t + seg_len - 0.1, 2),
                "text": "שלום עולם",
                "words": [
                    {"word": "שלום", "start": round(t, 2), "end": round(t + 1, 2), "probability": 0.95},
                    {"word": "עולם", "start": round(t + 1, 2), "end": round(t + seg_len - 0.1, 2), "probability": 0.9},
                ],
            }
        )
        t += seg_len
    return {"segments": segments}


def _make_entry(entry_dir: Path, duration_sec: float, source_id: str, entry_id: str) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    _write_sine_wav(entry_dir / "audio.wav", duration_sec)
    (entry_dir / "transcript.he.json").write_text(
        json.dumps(_make_transcript(duration_sec)), encoding="utf-8"
    )
    (entry_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_id": source_id,
                "source_entry_id": entry_id,
                "quality_score": 0.9,
                "license": "cc0",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_input(tmp_path: Path) -> Path:
    """A small input tree: a few entries of varying duration (incl. multi-slice)."""
    root = tmp_path / "input"
    # 5s -> 1 slice, 8s -> 1 slice, 65s -> multiple 30s slices (exercises prev_transcript)
    _make_entry(root / "e1", 5.0, "synthA", "e1")
    _make_entry(root / "e2", 8.0, "synthA", "e2")
    _make_entry(root / "e3", 65.0, "synthB", "e3")
    return root


def _assert_dataset_ok(ds):
    assert ds is not None
    assert ds.num_rows > 0

    # Schema present and complete (regression guard against dropped columns).
    feats = ds.features
    for col in ["audio", "transcript", "metadata", "has_prev", "has_timestamps", "prev_transcript"]:
        assert col in feats, f"missing column {col}"
    for mcol in ["seek", "duration", "source", "entry_id", "quality_score"]:
        assert mcol in feats["metadata"], f"missing metadata.{mcol}"

    # Audio feature must carry the target sampling rate (the cast/schema fix).
    assert feats["audio"].sampling_rate == SR

    # First row decodes to a real waveform at the right sampling rate.
    row = ds[0]
    assert row["audio"]["sampling_rate"] == SR
    assert len(row["audio"]["array"]) > 0
    assert isinstance(row["transcript"], str) and len(row["transcript"]) > 0
    assert "<|" in row["transcript"]  # timestamp tokens present

    # The multi-slice file must yield at least one row with prev conditioning.
    assert any(ds[i]["has_prev"] for i in range(ds.num_rows))


def test_prepare_single_proc(synthetic_input: Path):
    ds = prepare_training_dataset(
        input_folder=synthetic_input,
        slice_length=30,
        num_proc=1,
        per_proc_per_chunk_size=1,
    )
    _assert_dataset_ok(ds)


def test_prepare_multi_proc(synthetic_input: Path):
    """Parallel path with one-file-per-shard (mirrors production --per_proc_per_chunk_size 1)."""
    ds = prepare_training_dataset(
        input_folder=synthetic_input,
        slice_length=30,
        num_proc=2,
        per_proc_per_chunk_size=1,
    )
    _assert_dataset_ok(ds)


def test_exclude_filter_drops_source(synthetic_input: Path):
    ds = prepare_training_dataset(
        input_folder=synthetic_input,
        slice_length=30,
        num_proc=1,
        per_proc_per_chunk_size=1,
        exclude_filters=[("source_id", "eq", "synthB")],
    )
    assert ds is not None
    sources = {ds[i]["metadata"]["source"] for i in range(ds.num_rows)}
    assert "synthB" not in sources
    assert "synthA" in sources


def test_copy_metadata_fields(synthetic_input: Path):
    ds = prepare_training_dataset(
        input_folder=synthetic_input,
        slice_length=30,
        num_proc=1,
        per_proc_per_chunk_size=1,
        copy_metadata_fields=["license"],
    )
    assert ds is not None
    assert "license" in ds.features["metadata"]
    assert ds[0]["metadata"]["license"] == "cc0"
