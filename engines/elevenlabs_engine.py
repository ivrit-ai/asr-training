import io
import soundfile
import os
import time
from typing import Dict, Any, Callable, Tuple
from elevenlabs import ElevenLabs


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path", "scribe_v1")  # Default to scribe_v1 if not specified
    
    print('Initializing ElevenLabs client, model_path: ', model_path)

    # Initialize ElevenLabs client
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def transcribe(entry):
        # Convert audio to WAV format
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        try:
            # Call ElevenLabs Speech-to-Text API
            start_time = time.time()
            response = client.speech_to_text.convert(
                model_id=model_path,
                file=wav_buffer,
                language_code="heb",
                tag_audio_events=False
            )
            transcription_time = time.time() - start_time
            return response.text, transcription_time
        except Exception as e:
            print(f"Exception calling ElevenLabs API: {e}")
            raise e

    return transcribe 