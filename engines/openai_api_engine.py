import io
import soundfile
import pydub
import os
from openai import OpenAI
from typing import Dict, Any, Callable


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path", "whisper-1")  # Default to whisper-1 if not specified
    
    print('Initializing OpenAI client, model_path: ', model_path)

    # Initialize OpenAI client

    client = OpenAI()
    client.api_key = os.environ["OPENAI_API_KEY"]

    def transcribe(entry):
        # Convert audio to MP3 format
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        # Convert WAV to MP3
        audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)

        try:
            # Call OpenAI's Whisper API
            response = client.audio.transcriptions.create(
                model=model_path,
                file=("audio.mp3", mp3_buffer, "audio/mpeg"),
                language="he"  # Specify Hebrew language
            )
            return response.text
        except Exception as e:
            print(f"Exception calling OpenAI Whisper API: {e}")
            raise e

    return transcribe 