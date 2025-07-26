import io
import time
from typing import Any, Callable, Dict, Tuple, List
import librosa
import soundfile
import faster_whisper
import numpy as np


def get_device_and_index(device: str) -> tuple[str, int | None]:
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)
        return device, device_index

    return device, None


def transcribe_streaming(model, entry: Dict[str, Any]) -> Tuple[str, float]:
    # Resample audio to 16kHz
    audio_data = librosa.resample(
        entry["audio"]["array"], 
        orig_sr=entry["audio"]["sampling_rate"], 
        target_sr=16000
    )
    
    if len(audio_data) / 16000 < 0.5:
        return "", 0.0

    start_time = time.time()
    
    # Streaming parameters
    chunk_duration_ms = 500
    chunk_samples = int(16000 * chunk_duration_ms / 1000)  # 3200 samples for 200ms
    
    buffer = np.array([], dtype=np.float32)
    completed_segments = []
    cumulative_samples_removed = 0
    
    # Process audio in chunks
    for i in range(0, len(audio_data), chunk_samples):
        # Add new chunk to buffer
        chunk = audio_data[i:i + chunk_samples]
        buffer = np.concatenate([buffer, chunk])
        
        # Skip if buffer is too short
        if len(buffer) < chunk_samples:
            continue
            
        # Create audio buffer for faster-whisper
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, buffer, 16000, format="WAV")
        wav_buffer.seek(0)
        
        # Run faster-whisper on current buffer
        segments, _ = model.transcribe(wav_buffer, language="he")
        segments = list(segments)  # Convert generator to list
        
        # Calculate buffer start time in absolute audio timeline
        buffer_start_time = cumulative_samples_removed / 16000.0
        buffer_end_time = buffer_start_time + len(buffer) / 16000.0
        
        #print(f"Buffer: {buffer_start_time:.3f}s - {buffer_end_time:.3f}s (length: {len(buffer)/16000:.3f}s), segments={len(segments)}, {segments}")

        # If we have more than 1 segment, the first one is complete
        if len(segments) > 1:
            # Get the first complete segment
            first_segment = segments[0]
            completed_segments.append(first_segment.text.strip())
            
            # Calculate absolute timing
            absolute_start = buffer_start_time + first_segment.start
            absolute_end = buffer_start_time + first_segment.end
            
            # Calculate time delta between segment end and buffer length
            buffer_duration = len(buffer) / 16000.0
            time_delta = buffer_duration - first_segment.end
            
            #print(f"Segment: '{first_segment.text.strip()}'")
            #print(f"  Absolute timing: {absolute_start:.3f}s - {absolute_end:.3f}s")
            #print(f"  Relative timing: {first_segment.start:.3f}s - {first_segment.end:.3f}s")
            #print(f"  Time delta (buffer_end - segment_end): {time_delta:.3f}s")

            # Remove the completed segment from buffer based on its timing
            segment_end_time = first_segment.end
            samples_to_remove = int(segment_end_time * 16000)
            
            if samples_to_remove < len(buffer):
                buffer = buffer[samples_to_remove:]
                cumulative_samples_removed += samples_to_remove
            else:
                buffer = np.array([], dtype=np.float32)
                cumulative_samples_removed += len(buffer)
    
    # Process any remaining audio in buffer
    if len(buffer) > chunk_samples:
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, buffer, 16000, format="WAV")
        wav_buffer.seek(0)
        
        segments, _ = model.transcribe(wav_buffer, language="he")
        segments = list(segments)
        
        for segment in segments:
            completed_segments.append(segment.text.strip())
    
    transcription_time = time.time() - start_time
    return " ".join(completed_segments), transcription_time


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    device_index = None

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

    args = { 'device' : device }
    if device_index:
        args['device_index'] = device_index

    print(f'Loading faster-whisper streaming model: {model_path} on {device} with index: {device_index or 0}')
    model = faster_whisper.WhisperModel(model_path, **args)

    def transcribe_fn(entry):
        return transcribe_streaming(model, entry)

    return transcribe_fn 