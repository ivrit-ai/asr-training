import os
import threading
import time
from typing import Any, Callable, Dict, Tuple

import librosa
import numpy as np
import sherpa_onnx
import soundfile
from huggingface_hub import snapshot_download

DEFAULT_MODEL = "thewh1teagle/zipformer2-66m-streaming-he"
SAMPLE_RATE = 16_000
# Trailing silence so the last (partial) 1.28s chunk is flushed through the encoder.
TAIL_PADDING_SECONDS = 1.5


def _resolve_model_dir(model_path: str) -> str:
    if not os.path.isdir(model_path):
        model_path = snapshot_download(model_path, allow_patterns=["sherpa-onnx/*"])
    sherpa_dir = os.path.join(model_path, "sherpa-onnx")
    return sherpa_dir if os.path.isdir(sherpa_dir) else model_path


def _load_audio(entry: Dict[str, Any]) -> np.ndarray:
    audio = entry["audio"]
    if isinstance(audio, str):
        samples, sample_rate = soundfile.read(audio, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
    else:
        samples, sample_rate = np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
    # Resample up front (outside the timed section) rather than via sherpa-onnx's per-stream resampler.
    if sample_rate != SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return samples


def transcribe(recognizer: sherpa_onnx.OnlineRecognizer, entry: Dict[str, Any]) -> Tuple[str, float]:
    samples = _load_audio(entry)

    start_time = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    stream.accept_waveform(SAMPLE_RATE, np.zeros(int(TAIL_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream)
    transcription_time = time.time() - start_time

    return text.strip(), transcription_time


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path") or DEFAULT_MODEL
    device: str = kwargs.get("device", "auto")
    num_threads = int(kwargs.get("num_threads") or os.environ.get("SHERPA_ONNX_NUM_THREADS", 4))
    provider = "cuda" if device.startswith("cuda") else "cpu"

    model_dir = _resolve_model_dir(model_path)
    print(f"Loading sherpa-onnx streaming zipformer2 model: {model_dir} (provider={provider}, threads={num_threads})")

    def build_recognizer() -> sherpa_onnx.OnlineRecognizer:
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(model_dir, "tokens.txt"),
            encoder=os.path.join(model_dir, "encoder.int8.onnx"),
            decoder=os.path.join(model_dir, "decoder.int8.onnx"),
            joiner=os.path.join(model_dir, "joiner.int8.onnx"),
            model_type="zipformer2",
            num_threads=num_threads,
            provider=provider,
        )

    # One recognizer per worker thread, so --workers N runs N independent decoders.
    local = threading.local()
    build_recognizer()  # Fail fast on bad model paths before workers start.

    def transcribe_fn(entries):
        if not hasattr(local, "recognizer"):
            local.recognizer = build_recognizer()
        if not isinstance(entries, list):
            entries = [entries]
        return [transcribe(local.recognizer, entry) for entry in entries]

    return transcribe_fn
