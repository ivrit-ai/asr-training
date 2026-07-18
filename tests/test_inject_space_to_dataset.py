from inject_space_to_dataset import restore_timestamp_spaces


def test_restores_spaces_after_timestamp_tokens_before_text():
    transcripts = [
        "<|0.52|>ילד שנמצא באופקים, אם זה באשכול 4,<|3.42|><|3.98|>ההורה יקבל את הקייטנה בחינם.<|5.94|>",
    ]

    assert restore_timestamp_spaces(transcripts) == {
        "transcript": [
            "<|0.52|> ילד שנמצא באופקים, אם זה באשכול 4,<|3.42|><|3.98|> ההורה יקבל את הקייטנה בחינם.<|5.94|>",
        ]
    }


def test_preserves_existing_spaces_and_adjacent_timestamp_tokens():
    transcripts = ["<|0.00|> already spaced<|1.00|><|1.02|>still text<|2.00|>"]

    assert restore_timestamp_spaces(transcripts) == {
        "transcript": ["<|0.00|> already spaced<|1.00|><|1.02|> still text<|2.00|>"]
    }
