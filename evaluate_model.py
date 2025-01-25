#!/usr/bin/env python3

import argparse
import pickle

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faster_whisper
import boto3
import time
from google.cloud import speech, storage
import asyncio

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

import torch
import concurrent.futures

import io
import datasets
import jiwer
import json
import pandas as pd

import soundfile
import numpy
import pydub
import pandas

import whisper
import whisper.normalizers

import requests

import os

import uuid

from hebrew import Hebrew
import runpod
import base64

# Supported engines and models:
#
# 1. Engine: openai-whisper
#    - Models: large-v2, large-v3
# 2. Engine: transformers
#    - Models: openai/whisper-large-v2, openai/whisper-large-v3, user-trained models
# 3. Engine: faster-whisper
#    - Models: large-v2, large-v3, user-trained models
# 4. Engine: amazon-transcribe
#    - Models: batch, stream (uses AWS service)
# 5. Engine: google-speech
#    - Models: not applicable (uses Google Cloud service)
# 6. Engine: runpod:faster-whisper


def remove_niqqud(text: str):
    """Remove niqqud from Hebrew text."""
    return Hebrew(text).no_niqqud().string


def initialize_model(engine, model_path, tuned_model_path):
    if engine == "google-speech":
        speech_client = speech.SpeechClient()

        def transcribe(entry):
            return transcribe_google(speech_client, entry)

        return transcribe

    if engine == "amazon-transcribe":
        transcribe_client = boto3.client("transcribe")

        if model_path == "batch":
            # Initialize S3 bucket when batch mode is selected
            s3_client = boto3.client("s3")
            bucket_name = ensure_transcription_bucket(s3_client)

            def transcribe(entry):
                return transcribe_amazon_s3(transcribe_client, s3_client, bucket_name, entry)

        elif model_path == "stream":

            def transcribe(entry):
                return transcribe_amazon(transcribe_client, entry)

        else:
            raise ValueError("For amazon-transcribe, model must be 'stream' or 'batch'.")

        return transcribe

    if engine == "openai-whisper":
        model = whisper.load_model(model_path)

        def transcribe(entry):
            return transcribe_openai_whisper(model, entry)

        return transcribe

    if engine == "openai-whisper-tuned":
        model = whisper.load_model(model_path)

        if tuned_model_path:
            print(f"Loading tuned model {tuned_model_path}...")
            tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
            model = copy_hf_model_weights_to_openai_model_weights(tuned_model, model)

        def transcribe(entry):
            return transcribe_openai_whisper(model, entry)

        return transcribe

    if engine == "transformers":
        model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to("cuda:0")

        processor = WhisperProcessor.from_pretrained(model_path)

        def transcribe(entry):
            return transcribe_transformers(model, processor, entry)

        return transcribe

    if engine == "faster-whisper":
        model = faster_whisper.WhisperModel(model_path, device="cuda")

        def transcribe(entry):
            return transcribe_faster_whisper(model, entry)

        return transcribe

    if engine == "runpod:faster-whisper":
        # Initialize runpod with API key
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT"])

        def transcribe(entry):
            return transcribe_runpod_faster_whisper(endpoint, model_path, entry)

        return transcribe

    raise Exception


def transcribe_transformers(model, processor, entry):
    audio_resample = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)
    input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to("cuda:0")
    predicted_ids = model.generate(input_features, language="he", num_beams=5)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]


def transcribe_faster_whisper(model, entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)

    texts = []
    segs, dummy = model.transcribe(wav_buffer, language="he")
    for s in segs:
        texts.append(s.text)

    return " ".join(texts)


def copy_hf_model_weights_to_openai_model_weights(tuned_model, model):
    dic_parameter_mapping = dict()

    for i in range(32):
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.key.weight"] = (
            f"model.decoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.bias"] = f"model.decoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.weight"] = (
            f"model.decoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.bias"] = f"model.decoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.weight"] = (
            f"model.decoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.bias"] = f"model.decoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.weight"] = (
            f"model.decoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.bias"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.weight"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.key.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.bias"] = f"model.decoder.layers.{i}.fc1.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.weight"] = f"model.decoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.bias"] = f"model.decoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.weight"] = f"model.decoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.bias"] = f"model.decoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.weight"] = f"model.decoder.layers.{i}.final_layer_norm.weight"

        dic_parameter_mapping[f"encoder.blocks.{i}.attn.key.weight"] = (
            f"model.encoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.weight"] = (
            f"model.encoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.weight"] = (
            f"model.encoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.weight"] = (
            f"model.encoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.weight"] = f"model.encoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.bias"] = f"model.encoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.bias"] = f"model.encoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.bias"] = f"model.encoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.bias"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.weight"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.bias"] = f"model.encoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.bias"] = f"model.encoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.weight"] = f"model.encoder.layers.{i}.final_layer_norm.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.weight"] = f"model.encoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.bias"] = f"model.encoder.layers.{i}.fc1.bias"

    dic_parameter_mapping["encoder.conv1.bias"] = "model.encoder.conv1.bias"
    dic_parameter_mapping["encoder.conv1.weight"] = "model.encoder.conv1.weight"
    dic_parameter_mapping["encoder.conv2.bias"] = "model.encoder.conv2.bias"
    dic_parameter_mapping["encoder.conv2.weight"] = "model.encoder.conv2.weight"
    dic_parameter_mapping["encoder.ln_post.bias"] = "model.encoder.layer_norm.bias"
    dic_parameter_mapping["encoder.ln_post.weight"] = "model.encoder.layer_norm.weight"
    dic_parameter_mapping["encoder.positional_embedding"] = "model.encoder.embed_positions.weight"

    dic_parameter_mapping["decoder.ln.bias"] = "model.decoder.layer_norm.bias"
    dic_parameter_mapping["decoder.ln.weight"] = "model.decoder.layer_norm.weight"
    dic_parameter_mapping["decoder.positional_embedding"] = "model.decoder.embed_positions.weight"
    dic_parameter_mapping["decoder.token_embedding.weight"] = "model.decoder.embed_tokens.weight"

    model_state_dict = model.state_dict()
    tuned_model_state_dict = tuned_model.state_dict()

    for source_param, target_param in dic_parameter_mapping.items():
        model_state_dict[source_param] = tuned_model_state_dict[target_param]

    model.load_state_dict(model_state_dict)

    return model


def transcribe_openai_whisper(model, entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)  # Rewind to the start of the buffer

    audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
    audio.export("x.mp3", format="mp3")

    return model.transcribe("x.mp3", language="he", beam_size=5, best_of=5)["text"]


def process_entry(args):
    i, entry, transcribe_fn, text_column, normalizer = args

    raw_ref_text = entry[text_column]
    raw_eval_text = transcribe_fn(entry)

    ref_text = normalizer(raw_ref_text)
    eval_text = normalizer(raw_eval_text)

    entry_metrics = jiwer.process_words([ref_text], [eval_text])

    entry_data = {
        "id": i,
        "reference_text": raw_ref_text,
        "predicted_text": raw_eval_text,
        "norm_reference_text": ref_text,
        "norm_predicted_text": eval_text,
        "wer": entry_metrics.wer,
        "wil": entry_metrics.wil,
        "substitutions": entry_metrics.substitutions,
        "deletions": entry_metrics.deletions,
        "insertions": entry_metrics.insertions,
        "hits": entry_metrics.hits,
    }

    for key in entry.keys():
        if key not in ["audio", text_column]:
            entry_data[f"metadata_{key}"] = entry[key]

    print(
        f"Evaluated entry {i+1}, WER={entry_metrics.wer}, WIL={entry_metrics.wil}, ref_text={ref_text}, eval_text={eval_text}"
    )
    return entry_data


class HebrewTextNormalizer:
    def __init__(self):
        self.whisper_normalizer = whisper.normalizers.BasicTextNormalizer()

    def __call__(self, text):
        # First remove niqqud, then apply whisper normalization
        text = remove_niqqud(text)
        text = text.replace('"', "").replace("'", "")

        return self.whisper_normalizer(text)


def evaluate_model(transcribe_fn, ds, text_column, num_workers=1):
    # Replace the basic normalizer with our custom Hebrew normalizer
    normalizer = HebrewTextNormalizer()

    entries_data = []

    # Prepare arguments for parallel processing
    process_args = [(i, ds[i], transcribe_fn, text_column, normalizer) for i in range(len(ds))]

    # Process entries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        entries_data = list(executor.map(process_entry, process_args))

    # Sort results by ID to maintain original order
    entries_data.sort(key=lambda x: x["id"])

    # Calculate final metrics
    final_metrics = jiwer.process_words(
        [entry["norm_reference_text"] for entry in entries_data],
        [entry["norm_predicted_text"] for entry in entries_data],
    )

    results_df = pandas.DataFrame(entries_data)

    return final_metrics, results_df


def transcribe_amazon(transcribe_client, entry):
    # Convert audio to proper format
    audio_data = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)

    audio_length = len(audio_data) / 16000
    if audio_length < 0.5:
        return ""

    # Convert to bytes
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, audio_data, 16000, format="WAV")
    audio_bytes = wav_buffer.getvalue()

    async def process_audio():
        client = TranscribeStreamingClient(region="us-west-2")

        stream = await client.start_stream_transcription(
            language_code="he-IL",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        # Handle audio chunks
        async def write_chunks():
            chunk_size = 1024 * 16
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()

        # Handle transcription results
        transcript = []

        class EventHandler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                results = transcript_event.transcript.results
                for result in results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            transcript.append(alt.transcript)

        # Process audio and get transcription
        handler = EventHandler(stream.output_stream)
        await asyncio.gather(write_chunks(), handler.handle_events())

        return " ".join(transcript)

    # Run async code
    return asyncio.run(process_audio())


def transcribe_google(speech_client, entry):
    # Convert audio to proper format
    audio_data = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)

    # Convert to bytes and save to temporary file
    temp_filename = f"/tmp/{uuid.uuid4()}.wav"
    soundfile.write(temp_filename, audio_data, 16000, format="WAV")

    # Use default credentials from service account
    storage_client = storage.Client()
    bucket_name = "stt-evaluation-audio-bucket"

    try:
        bucket = storage_client.get_bucket(bucket_name)
    except:
        bucket = storage_client.create_bucket(bucket_name)

        # Set lifecycle policy to expire objects after 2 days
        bucket.lifecycle_rules = [{"action": {"type": "Delete"}, "condition": {"age": 2}}]
        bucket.update()

    blob_name = f"audio/{uuid.uuid4()}.wav"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(temp_filename)

    # Clean up temp file
    os.remove(temp_filename)

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="he-IL",  # Hebrew
        model="default",
    )

    # Use long running recognize with service account auth
    operation = speech_client.long_running_recognize(config=config, audio=audio)

    # Wait for operation to complete
    response = operation.result(timeout=90)

    # Clean up GCS file
    blob.delete()

    # Combine all transcriptions
    transcript = " ".join(result.alternatives[0].transcript for result in response.results)

    return transcript


def ensure_transcription_bucket(s3_client):
    bucket_name = "stt-evaluation-transcription-bucket"

    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        # Create the bucket if it doesn't exist
        s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": s3_client.meta.region_name}
        )

        # Set lifecycle policy to expire objects after 2 days
        lifecycle_policy = {
            "Rules": [{"ID": "ExpireObjectsAfter2Days", "Prefix": "", "Status": "Enabled", "Expiration": {"Days": 2}}]
        }
        s3_client.put_bucket_lifecycle_configuration(Bucket=bucket_name, LifecycleConfiguration=lifecycle_policy)

    return bucket_name


def transcribe_amazon_s3(transcribe_client, s3_client, bucket_name, entry):
    # Convert audio to proper format
    audio_data = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)

    audio_length = len(audio_data) / 16000
    if audio_length < 0.5:
        return ""

    # Convert to bytes
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, audio_data, 16000, format="WAV")
    audio_bytes = wav_buffer.getvalue()

    # Upload to existing bucket with unique object key
    object_key = f"audio/{uuid.uuid4()}.wav"
    s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=audio_bytes)

    # Start transcription job
    job_name = f"transcription-job-{uuid.uuid4()}"
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{bucket_name}/{object_key}"},
        MediaFormat="wav",
        LanguageCode="he-IL",
    )

    # Wait for the job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        time.sleep(5)

    # Get the transcription result
    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
        transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        response = requests.get(transcript_uri)
        transcript = response.json()["results"]["transcripts"][0]["transcript"]
        return transcript
    else:
        print(status)
        raise Exception(f"Transcription job failed for entry {entry}")


def transcribe_runpod_faster_whisper(endpoint, model, entry):
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
    payload = {"type": "blob", "data": data, "model": model}

    # Call runpod endpoint
    try:
        result = endpoint.run_sync(payload)
        texts = [e["text"] for e in result[0]["result"]]
        return "".join(texts)
    except Exception as e:
        print(f"Exception calling runpod: {e}")
        return ""


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Create a dataset and upload to Huggingface.")

    # Add the arguments
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices={
            "openai-whisper",
            "openai-whisper-tuned",
            "transformers",
            "faster-whisper",
            "amazon-transcribe",
            "google-speech",
            "runpod:faster-whisper",
        },
        help="Engine to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use. Can be remote (e.g. openai/whisper-large-v3) or local (full path). For amazon-transcribe, use 'stream' or 'batch'.",
    )
    parser.add_argument("--tuned-model", type=str, required=False)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to evaluate in format dataset_name:<split>:<text_column>."
    )
    parser.add_argument("--name", type=str, required=False, help="Optional name parameter for dataset.load_dataset.")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="evaluation_results.csv",
        help="Output CSV file path. If not provided, will use 'evaluation_results.csv'",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers to use for evaluation")

    # Parse the arguments
    args = parser.parse_args()

    print(f"Loading engine {args.engine} with model {args.model}...")
    transcribe_fn = initialize_model(args.engine, args.model, args.tuned_model)

    print(f"Loading dataset {args.dataset}...")
    dataset_parts = args.dataset.split(":")
    dataset_name = dataset_parts[0]
    dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
    ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

    if args.name:
        ds = datasets.load_dataset(dataset_name, name=args.name)[dataset_split]
    else:
        ds = datasets.load_dataset(dataset_name)[dataset_split]

    print(f"Beginning evaluation with {args.workers} workers.")
    metrics, results_df = evaluate_model(transcribe_fn, ds, ds_text_column, args.workers)

    print(f"Evaluation done. WER={metrics.wer}, WIL={metrics.wil}.")

    # Add model and dataset info as columns
    results_df["model"] = args.model
    results_df["dataset"] = dataset_name
    results_df["dataset_split"] = dataset_split
    results_df["engine"] = args.engine

    output_file = args.output

    results_df.to_csv(output_file, encoding="utf-8", index=False)

    print(f"Results saved to {output_file}")
