import time
from typing import Callable, Tuple

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

    def transcribe(entries):
        if not isinstance(entries, list):
            entries = [entries]

        audio_resample = []
        for entry in entries:
            single_audio_resample = librosa.resample(
                entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
            )
            audio_resample.append(single_audio_resample)

        input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(model.device)

        start_time = time.time()
        predicted_ids = model.generate(input_features, language="he", num_beams=5)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription_time = time.time() - start_time

        return [(single_transcription, transcription_time / len(transcription)) for single_transcription in transcription]

    return transcribe
