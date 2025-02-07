import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, Any, Callable
import anthropic

SYSTEM_PROMPT = """You are an expert in Hebrew transcription post-processing.
Your task is to improve the transcription by:
1. Fixing common ASR errors
2. Correcting grammar and syntax
3. Maintaining the original meaning
4. Keeping the style natural and fluent

Only make necessary changes. If the transcription seems correct, return it as is.
Wrap your response in <improved> tags, like this:
<improved>your improved text here</improved>

Respond with the improved text in XML tags only, no explanations."""


def extract_improved_text(text: str) -> str:
    """Extract text between <improved> tags."""
    match = re.search(r"<improved>(.*?)</improved>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return original if no tags found


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model")
    base_engine_path = kwargs.get("base_engine", "engines/faster_whisper_engine.py")
    llm_model = kwargs.get("llm_model", "claude-3-opus-20240229")

    # Import the base engine
    base_engine_path = Path(base_engine_path)
    if not base_engine_path.exists():
        raise FileNotFoundError(f"Base engine not found: {base_engine_path}")

    spec = importlib.util.spec_from_file_location("base_engine", base_engine_path)
    base_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_engine)

    # Initialize the base transcription function
    transcribe_fn = base_engine.create_app(model=model_path)

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def improve_transcription(text: str) -> str:
        if not text.strip():
            return text

        try:
            response = client.messages.create(
                model=llm_model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=1000,
            )
            return extract_improved_text(response.content[0].text)
        except Exception as e:
            print(f"LLM processing failed: {e}")
            return text  # Return original text if LLM processing fails

    def transcribe(entry):
        # Get base transcription
        base_transcription = transcribe_fn(entry)

        # Improve using LLM
        improved_transcription = improve_transcription(base_transcription)

        return improved_transcription

    return transcribe
