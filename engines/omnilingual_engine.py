import io
import os
import tempfile
import time
from typing import Any, Callable, Dict, Tuple

import soundfile

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def transcribe(pipeline: ASRInferencePipeline, entry: Dict[str, Any], lang: str) -> Tuple[str, float]:
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_path = temp_wav.name
        soundfile.write(temp_path, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    
    try:
        start_time = time.time()
        transcriptions = pipeline.transcribe([temp_path], lang=[lang], batch_size=1)
        transcription_time = time.time() - start_time
        
        # Extract the transcription text from the result
        transcription_text = transcriptions[0] if transcriptions else ""
        
        return transcription_text, transcription_time
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def create_app(**kwargs) -> Callable:
    model_card = kwargs.get("model_card", "omniASR_LLM_7B")
    lang = kwargs.get("lang", "heb_Hebr")
    
    print(f'Loading omnilingual-asr model: {model_card} for language: {lang}')
    pipeline = ASRInferencePipeline(model_card=model_card)
    
    def transcribe_fn(entry):
        return transcribe(pipeline, entry, lang)
    
    return transcribe_fn

