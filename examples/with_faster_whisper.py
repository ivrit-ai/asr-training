"""
Example using faster-whisper with ctranslate2 backend for fast audio transcription.
Run with:
    pip install faster-whisper
    python with_faster_whisper.py
    
Usage on Ubuntu 22.04 with CUDA:
1. Use python 3.11
2. Install CudNN 9.1.0 with:
    pip install faster-whisper==1.1.1
    wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb && sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update && sudo apt-get -y install cudnn -y
"""

import faster_whisper
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

segs, _ = model.transcribe('audio.opus', language='he')
text = ' '.join(s.text for s in segs)
print(f'Transcribed text: {text}')
