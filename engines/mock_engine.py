from typing import Callable


def create_app(**kwargs) -> Callable:

    def transcribe(entry):
        return entry[kwargs["text_column"]]

    return transcribe
