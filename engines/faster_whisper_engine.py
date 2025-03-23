import io
from typing import Any, Callable, Dict

import faster_whisper
import soundfile


def transcribe(model, entry: Dict[str, Any]) -> str:
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)

    texts = []
    segs, dummy = model.transcribe(wav_buffer, language="he")
    for s in segs:
        texts.append(s.text)

    return " ".join(texts)


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    device_index = None
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)

    model = faster_whisper.WhisperModel(model_path, device=device, device_index=device_index)

    def transcribe_fn(entry):
        return transcribe(model, entry)

    return transcribe_fn
