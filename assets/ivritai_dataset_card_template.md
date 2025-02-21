---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for {{ pretty_name | default("Dataset Name", true) }}

<!-- Provide a quick summary of the dataset. -->

{{ dataset_summary | default("", true) }}

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

{{ dataset_description | default("", true) }}

- **License:** {{ license | default("[More Information Needed]", true)}}

## Dataset Structure

### Data Fields

Each example in the dataset contains:

- `audio`: An audio column containing:
  - `bytes`: The audio data encoded in MP3 format
  - `path`: A string identifier derived from the source entry ID
  - Sampling rate: Fixed at 16000 Hz
- `transcript`: A string containing the text with potentially Whisper-style timestamp tokens (e.g., `<|0.00|>text<|2.40|>`) if "has_timestamps" is true
- `metadata`: A dictionary containing:
  - `seek`: Float indicating the start time of this slice in the original source audio
  - `source`: String identifier for the source of the audio (Name of podcast, production system, etc.)
  - `entry_id`: Unique identifier for the source entry
- `has_prev`: Boolean indicating if this slice has transcript from the previous slice within the audio source
- `has_timestamps`: Boolean indicating if the transcript contains timestamp tokens
- `prev_transcript`: String containing the transcript of the previous slice (empty if `has_prev` is false)