import io
import soundfile
from typing import Dict, Any, Callable
import faster_whisper


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
    model = faster_whisper.WhisperModel(model_path)

    def transcribe_fn(entry):
        return transcribe(model, entry)

    return transcribe_fn
