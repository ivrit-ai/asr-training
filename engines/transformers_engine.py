import torch
import librosa
from typing import Dict, Any, Callable
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")

    torch_dtype = torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )

    def transcribe(entry):
        audio_resample = librosa.resample(
            entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
        )
        generate_kwargs = {
            "max_new_tokens": 224,
            "num_beams": 5,
            "condition_on_prev_tokens": True,
            "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
            "language": "hebrew",
        }

        result = pipe(audio_resample, generate_kwargs=generate_kwargs)

        return result["text"]

    return transcribe
