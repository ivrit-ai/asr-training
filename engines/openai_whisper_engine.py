import io
import os
from typing import Callable

import pydub
import soundfile
import whisper
from transformers import WhisperForConditionalGeneration


def copy_hf_model_weights_to_openai_model_weights(tuned_model, model):
    dic_parameter_mapping = dict()

    for i in range(32):
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.key.weight"] = (
            f"model.decoder.layers.{i}.self_attn.k_proj.weight"
        )
        # ... (rest of the parameter mapping code)
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.bias"] = f"model.encoder.layers.{i}.fc1.bias"

    # Other mappings
    dic_parameter_mapping["encoder.conv1.bias"] = "model.encoder.conv1.bias"
    # ... (rest of the parameter mapping code)
    dic_parameter_mapping["decoder.token_embedding.weight"] = "model.decoder.embed_tokens.weight"

    model_state_dict = model.state_dict()
    tuned_model_state_dict = tuned_model.state_dict()

    for source_param, target_param in dic_parameter_mapping.items():
        model_state_dict[source_param] = tuned_model_state_dict[target_param]

    model.load_state_dict(model_state_dict)
    return model


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    tuned_model_path = kwargs.get("tuned_model_path")

    model = whisper.load_model(model_path)

    if tuned_model_path:
        print(f"Loading tuned model {tuned_model_path}...")
        tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
        model = copy_hf_model_weights_to_openai_model_weights(tuned_model, model)

    device: str = kwargs.get("device", None)
    if device:
        model.to(device)

    def transcribe(entry):
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
        audio.export("temp.mp3", format="mp3")

        result = model.transcribe("temp.mp3", language="he", beam_size=5, best_of=5)
        os.remove("temp.mp3")

        return result["text"]

    return transcribe
