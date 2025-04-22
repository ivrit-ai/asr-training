"""
Example of using whispercpp for fast and lightweight transcription.
Download the model from Hugging Face:
    wget https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin
Run with:
    pip install pywhispercpp huggingface-hub
    python with_whisper_cpp.py
"""

from pywhispercpp.model import Model
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="ivrit-ai/whisper-large-v3-turbo-ggml",
    filename="ggml-model.bin"
)
model = Model(model_path)
segs = model.transcribe('audio.opus', language='he')
text = ' '.join(segment.text for segment in segs)
print(f'Transcribed text: {text}')