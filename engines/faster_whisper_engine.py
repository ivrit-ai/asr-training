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


def get_device_and_index(device: str) -> tuple[str, int | None]:
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)
        return device, device_index

    return device, None


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    device_index = None

    if len(device.split(",")) > 1:
        device_indexes = []
        base_device = None
        for device_instance in device.split(","):
            device, device_index = get_device_and_index(device_instance)
            base_device = base_device or device
            if base_device != device:
                raise ValueError("Multiple devices must be instances of the same base device (e.g cuda:0, cuda:1 etc.)")
            device_indexes.append(device_index)
        device = base_device
        device_index = device_indexes
    else:
        device, device_index = get_device_and_index(device)

    print(f'Loading faster-whisper model: {model_path} on {device} with index: {device_index or "auto"}')
    model = faster_whisper.WhisperModel(model_path, device=device, device_index=device_index)

    def transcribe_fn(entry):
        return transcribe(model, entry)

    return transcribe_fn
