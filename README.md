# Training

ASR training recipes created for ivrit.ai
This is not yet properly documented - Soon to come.

# Evaluation

## Running the entire evaluation bench

The `run_bench.py` will run the suite of eval datasets using a specified engine.

Example command line is:

```bash
python run_bench.py \
    --engine engines/faster_whisper_engine.py \
    --model /path-to-your-ct2-whisper-model \
    --output-dir /path-to-eval-result-csvs \
    --overwrite
```

This will use the "faster-whisper" engine, so a CT2 model is required.
Other engines include:

- HF Transformers
- Remote runpod container running faster-whisper
- OpenAI Whisper API
- Amazon Transcribe API
- Google Speech API

The above are not yet documented but feel free to look at the code on how to run them.

## Datasets in the suite

The following datasets are part of the evaluation suite:

| Label             | Dataset                                                                                                      | Split      | Text Column   | Dataset Configuration Name | Gated |
| ----------------- | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------- | -------------------------- | ----- |
| ivrit_ai_eval_d1  | [ivrit-ai/eval-d1](https://huggingface.co/datasets/ivrit-ai/eval-d1)                                         | test       | text          | -                          | ❌    |
| saspeech          | [upai-inc/saspeech](https://huggingface.co/datasets/ivrit-ai/saspeech)                                       | test       | text          | -                          | ❌    |
| fleurs            | [google/fleurs](https://huggingface.co/datasets/google/fleurs)                                               | test       | transcription | he_il                      | ❌    |
| common_voice_17   | [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | test       | sentence      | he                         | ✅    |
| hebrew_speech_kan | [imvladikon/hebrew_speech_kan](https://huggingface.co/datasets/imvladikon/hebrew_speech_kan)                 | validation | sentence      | -                          | ❌    |

## Leaderboard

We publish the results on the [HF Leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard).
Contact us to add your model to the leaderboard.

## Common Problems

## `Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}`

If the evaluation engine `faster-whisper` is used the following error may show up.

The CTranslate 2 engine depends on CUDNN to run on Nvidia GPUs. This library is actually installed already using pip but is not on the dynamic library path most likely.
The following line will put it on path for the current session. If you use a virtual env - make sure it's active before running it so it can infer the correct pip folder.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
