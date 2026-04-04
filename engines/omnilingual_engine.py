import os
import tempfile
import time
from typing import Any, Callable, Dict, Tuple

import soundfile

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaBeamSearchConfig


def transcribe_batch(pipeline: ASRInferencePipeline, entries: list[Dict[str, Any]], lang: str) -> list[Tuple[str, float]]:
    temp_paths = []
    for entry in entries:
        # Handle audio loading if it's a path string
        if isinstance(entry["audio"], str):
            audio_array, sampling_rate = soundfile.read(entry["audio"])
            entry["audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            soundfile.write(temp_path, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
            temp_paths.append(temp_path)
    
    try:
        start_time = time.time()
        langs = [lang] * len(temp_paths)
        transcriptions = pipeline.transcribe(temp_paths, lang=langs, batch_size=len(temp_paths))
        transcription_time = time.time() - start_time
        results = [(t, transcription_time / len(temp_paths)) for t in transcriptions]
        return results
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)


def create_app(**kwargs) -> Callable:
    model_card = kwargs.get("model_path", "omniASR_LLM_Unlimited_7B_v2") # model path is the model card name
    lang = kwargs.get("lang", "heb_Hebr")

    device: str = kwargs.get("device", "cuda")

    print(f'Loading omnilingual-asr model: {model_card} for language: {lang} on device: {device}')
    beam_search_config = Wav2Vec2LlamaBeamSearchConfig(nbest = 5)
    pipeline = ASRInferencePipeline(model_card=model_card, device=device, beam_search_config=beam_search_config)
    
    def transcribe_fn(entry):
        if not isinstance(entry, list):
            entries = [entry]
            was_single = True
        else:
            entries = entry
            was_single = False
        
        results = transcribe_batch(pipeline, entries, lang)
        
        if was_single:
            return results[0]
        else:
            return results
    
    return transcribe_fn

