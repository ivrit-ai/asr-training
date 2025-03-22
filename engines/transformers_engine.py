from typing import Callable

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
    model.to(device)
    processor = WhisperProcessor.from_pretrained(model_path)

    def transcribe(entry):
        audio_resample = librosa.resample(
            entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
        )
        input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(model.device)

        predicted_ids = model.generate(input_features, language="he", num_beams=5)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0]

    return transcribe
