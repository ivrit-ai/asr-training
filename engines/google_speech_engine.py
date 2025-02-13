import io
import soundfile
import uuid
import os
import librosa
from typing import Dict, Any, Callable
from google.cloud import speech, storage


def create_app(**kwargs) -> Callable:
    speech_client = speech.SpeechClient()
    storage_client = storage.Client()
    bucket_name = "stt-evaluation-audio-bucket"

    # Ensure bucket exists
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except:
        bucket = storage_client.create_bucket(bucket_name)
        bucket.lifecycle_rules = [{"action": {"type": "Delete"}, "condition": {"age": 2}}]
        bucket.update()

    def transcribe(entry):
        # Convert audio to proper format
        audio_data = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)

        # Save to temporary file
        temp_filename = f"/tmp/{uuid.uuid4()}.wav"
        soundfile.write(temp_filename, audio_data, 16000, format="WAV")

        # Upload to GCS
        blob_name = f"audio/{uuid.uuid4()}.wav"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_filename)
        os.remove(temp_filename)

        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="he-IL",
            model="default",
        )

        operation = speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=90)
        blob.delete()

        return " ".join(result.alternatives[0].transcript for result in response.results)

    return transcribe
