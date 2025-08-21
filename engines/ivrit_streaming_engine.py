import io
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import ivrit


def transcribe_streaming(model, entry: Dict[str, Any], chunk_duration_in_seconds: float = 1.0) -> Tuple[str, float]:
    """Transcribe audio using Ivrit streaming session."""
    # Convert audio to 16-bit PCM S16LE format
    
    audio_array = entry["audio"]["array"]
    sampling_rate = entry["audio"]["sampling_rate"]
    
    # Ensure audio is in the correct format (float32 normalized to [-1, 1])
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Convert to 16-bit PCM S16LE (little endian)
    # Scale from [-1, 1] to [-32768, 32767] and convert to int16
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Convert to bytes (S16LE format)
    audio_data = audio_int16.tobytes()
    
    start_time = time.time()
    
    # Create streaming session
    session = model.create_session()
    
    # Calculate chunk size based on duration and sampling rate
    # Each sample is 2 bytes (16-bit) and we have sampling_rate samples per second
    samples_per_chunk = int(sampling_rate * chunk_duration_in_seconds)
    bytes_per_sample = 2  # 16-bit audio
    chunk_size = samples_per_chunk * bytes_per_sample
    
    # Append audio data in chunks
    print(f'Appending audio data in chunks of {chunk_size} bytes')
    for i in range(0, len(audio_data), chunk_size):
        print(f'Appending audio data in chunk {i} of {len(audio_data)} bytes')
        chunk = audio_data[i:i + chunk_size]
        session.append(chunk)
    
    # Flush to get final results
    session.flush()
    
    transcript = session.get_full_text()
    
    transcription_time = time.time() - start_time
    
    return transcript, transcription_time


def get_device_and_index(device: str) -> tuple[str, int | None]:
    """Parse device string to extract device type and index."""
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)
        return device, device_index

    return device, None


def create_app(**kwargs) -> Callable:
    """Create the Ivrit streaming transcription application."""
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    chunk_duration_in_seconds: float = kwargs.get("chunk_duration_in_seconds", 15.0)
    device_index = None

    if not model_path:
        raise ValueError("model_path is required for ivrit streaming engine")

    # Handle multiple devices (similar to faster_whisper_engine)
    if len(device.split(",")) > 1:
        device_indexes = []
        base_device = None
        for device_instance in device.split(","):
            device, device_index = get_device_and_index(device_instance)
            base_device = base_device or device
            if base_device != device:
                raise ValueError("Multiple devices must be instances of the same base device (e.g cuda:0, cuda:1 etc.)")
            device_indexes.append(device_index)
        device = base_device
        device_index = device_indexes
    else:
        device, device_index = get_device_and_index(device)

    # Prepare arguments for ivrit.load_model
    load_args = {'engine': 'faster-whisper', 'model': model_path}
    
    # Add device configuration if specified
    if device != "auto":
        load_args['device'] = device
        if device_index is not None:
            load_args['device_index'] = device_index

    print(f'Loading ivrit streaming model: {model_path} on {device} with index: {device_index or 0}, chunk_duration: {chunk_duration_in_seconds}s')
    
    # Load the model using ivrit
    model = ivrit.load_model(**load_args)

    def transcribe_fn(entry):
        return transcribe_streaming(model, entry, chunk_duration_in_seconds)

    return transcribe_fn
