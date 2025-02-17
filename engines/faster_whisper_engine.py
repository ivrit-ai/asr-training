import io
import soundfile
from typing import Dict, Any, Callable
import faster_whisper

import torch

has_cuda = torch.cuda.is_available()


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
    model = kwargs.get("model")
    device = "cuda" if has_cuda else "auto"
    compute_type = "float16" if has_cuda else "default"
    print(f"Initializing model {model}...")
    model = faster_whisper.WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe_fn(entry):
        return transcribe(model, entry)

    return transcribe_fn
