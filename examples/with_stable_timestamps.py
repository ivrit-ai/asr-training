"""
Example of Stable-ts with faster-whisper for fast and accurate transcription.
Run with:
    pip install -U 'stable-ts[fw]'
    python with_stable_timestamps.py
"""

import stable_whisper

model = stable_whisper.load_faster_whisper('ivrit-ai/whisper-large-v3-turbo-ct2')
segs = model.transcribe('audio.opus', language='he') # Word level timestamps enabled by default
for s in segs:
    print(f'{s.start:.2f} - {s.end:.2f}: {s.text}')