import io
import soundfile
import pydub
import base64
import os
import time
import runpod
from typing import Dict, Any, Callable, Tuple


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")

    # Initialize runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT"])

    def transcribe(entry):
        # Convert audio to MP3 format
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        # Convert WAV to MP3
        audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3")
        mp3_data = mp3_buffer.getvalue()

        # Encode MP3 data to base64
        data = base64.b64encode(mp3_data).decode("utf-8")

        # Prepare payload
        payload = {"type": "blob", "data": data, "model": model_path}

        try:
            start_time = time.time()
            result = endpoint.run_sync(payload)
            transcription_time = time.time() - start_time
            texts = [e["text"] for e in result[0]["result"]]
            return " ".join(texts), transcription_time
        except Exception as e:
            print(f"Exception calling runpod: {e}")
            raise e

    return transcribe
