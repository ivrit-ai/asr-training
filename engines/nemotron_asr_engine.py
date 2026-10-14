import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import ivrit
import soundfile

TARGET_LANG = "he"


def transcribe(model, entry: Dict[str, Any]) -> Tuple[str, float]:
    fd, wav_path = tempfile.mkstemp(prefix="nemo_asr_", suffix=".wav")
    os.close(fd)
    try:
        soundfile.write(wav_path, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")

        start_time = time.time()
        result = model.transcribe(path=wav_path, language=TARGET_LANG)
        transcription_time = time.time() - start_time
    finally:
        Path(wav_path).unlink(missing_ok=True)

    return result["text"].strip(), transcription_time


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path") or "nvidia/nemotron-3.5-asr-streaming-0.6b"
    device: str = kwargs.get("device", "auto")

    print(f"Loading nemotron-asr model: {model_path} on {device}")
    model = ivrit.load_model(engine="nemotron-asr", model=model_path, device=device)

    def transcribe_fn(entries):
        if not isinstance(entries, list):
            entries = [entries]
        return [transcribe(model, entry) for entry in entries]

    return transcribe_fn
