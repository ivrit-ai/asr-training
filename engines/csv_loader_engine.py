import pandas as pd
from typing import Dict, Any, Callable


def create_app(**kwargs) -> Callable:
    """Create a function that returns predictions from a CSV file.

    Args:
        model: Path to the CSV file containing predictions
    """
    csv_path = kwargs.get("model")

    # Load the CSV file
    print(f"Loading predictions from {csv_path}")
    df = pd.read_csv(csv_path)

    if "predicted_text" not in df.columns:
        raise ValueError(f"Column 'predicted_text' not found in {csv_path}")

    # Create a mapping from audio file to prediction
    predictions = df["predicted_text"].tolist()
    current_idx = 0

    def transcribe(entry) -> str:
        nonlocal current_idx
        if current_idx >= len(predictions):
            raise IndexError(f"No more predictions available in {csv_path}")

        prediction = predictions[current_idx]
        current_idx += 1
        return prediction if isinstance(prediction, str) else ""

    return transcribe
