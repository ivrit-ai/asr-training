import pytest
from stable_whisper.result import Segment, WordTiming

from create_dataset import generate_slices

test_cases = [
    pytest.param(
        [
            Segment(0, 0, "Zero"),
            Segment(15, 20, "Hello"),
        ],
        30,
        [
            {
                "seek": 15,
                "segments": [{"start": 0, "end": 5, "text": "Hello", "word_scores": []}],
            }
        ],
        0,
        id="first_zero_len_segment_skipped",
    ),
    pytest.param(
        [
            Segment(15, 20, "Hello"),
            Segment(40, 40, "Zero"),
        ],
        30,
        [
            {
                "seek": 0,
                "segments": [{"start": 15, "end": 20, "text": "Hello", "word_scores": []}],
            },
        ],
        0,
        id="last_zero_len_segment_skipped",
    ),
    pytest.param(
        [
            Segment(15, 20, "Hello"),
            Segment(35, 35, "Zero"),
            Segment(40, 45, "World"),
        ],
        45,
        [
            {
                "seek": 0,
                "segments": [{"start": 15, "end": 20, "text": "Hello", "word_scores": []}],
            },
            {
                "seek": 40,
                "segments": [{"start": 0, "end": 5, "text": "World", "word_scores": []}],
            },
        ],
        0,
        id="mid_zero_len_segment_skipped",
    ),
    pytest.param(
        [
            Segment(0, 10, "Hello"),
            Segment(15, 20, "World"),
        ],
        30,
        [
            {
                "seek": 0,
                "segments": [
                    {"start": 0, "end": 10, "text": "Hello", "word_scores": []},
                    {"start": 15, "end": 20, "text": "World", "word_scores": []},
                ],
            }
        ],
        0,
        id="basic_two_segments",
    ),
    pytest.param(
        [
            Segment(0, 10, "Hello"),
            Segment(15, 35, "World"),
        ],
        40,
        [
            {"seek": 0, "segments": [{"start": 0, "end": 10, "text": "Hello", "word_scores": []}, {"start": 15}]},
            {"seek": 10, "segments": [{"start": 5, "end": 25, "text": "World", "word_scores": []}]},
        ],
        0,
        id="last_segment_cross_over",
    ),
    pytest.param(
        [
            Segment(0, 10, "Hello"),
            Segment(50, 70, "World"),
        ],
        70,
        [
            {"seek": 0, "segments": [{"start": 0, "end": 10, "text": "Hello", "word_scores": []}]},
            {"seek": 30, "segments": []},
            {"seek": 50, "segments": [{"start": 0, "end": 20, "text": "World", "word_scores": []}]},
        ],
        0,
        id="last_segment_cross_over_single_crossed_over",
    ),
    pytest.param(
        [
            Segment(0, 35, "Hello"),
        ],
        30,
        [],
        0,
        id="segment_end_over_audio_duration",
    ),
    pytest.param(
        [
            Segment(0, 5, "Hello"),
        ],
        10,
        [
            {"seek": 0, "segments": [{"start": 0, "end": 5, "text": "Hello", "word_scores": []}]},
        ],
        0,
        id="basic_single_segment_in_a_single_slice",
    ),
    pytest.param(
        [
            Segment(35, 45, "Hello"),
        ],
        45,
        [
            {"seek": 0, "segments": []},
            {"seek": 30, "segments": [{"start": 5, "end": 15, "text": "Hello", "word_scores": []}]},
        ],
        0,
        id="first_slice_empty",
    ),
    pytest.param(
        [
            Segment(5, 15, "Hello"),
        ],
        45,
        [
            {"seek": 0, "segments": [{"start": 5, "end": 15, "text": "Hello", "word_scores": []}]},
            {"seek": 30, "segments": []},
        ],
        0,
        id="last_slice_empty",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 15, 0.9)]),
            Segment(words=[WordTiming("low", 20, 35, 0.1)]),
        ],
        45,
        [
            {"seek": 0, "segments": []},
        ],
        0.8,
        id="low_quality_slice_dropped_last_seg_low_quality",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("low", 5, 15, 0.1)]),
            Segment(words=[WordTiming("Hello", 20, 35, 0.9)]),
        ],
        45,
        [
            {"seek": 0, "segments": []},
            {"seek": 20, "segments": [{"start": 0, "end": 15, "text": "Hello", "word_scores": [0.9]}]},
        ],
        0.8,
        id="low_quality_slice_dropped_first_seg_low_quality",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 10, 0.8)]),
            Segment(words=[WordTiming("low", 15, 20, 0.1)]),
            Segment(words=[WordTiming("low", 20, 21, 0.1)]),
            Segment(words=[WordTiming("low", 21, 22, 0.1)]),
            Segment(words=[WordTiming("low", 22, 23, 0.1)]),
            Segment(words=[WordTiming("World", 25, 30, 0.8)]),
        ],
        30,
        [
            {"seek": 0, "segments": []},
            {"seek": 25, "segments": [{"start": 0, "end": 5, "text": "World", "word_scores": [0.8]}]},
        ],
        0.8,
        id="low_quality_slice_dropped_mid_seg_low_quality",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 10, 0.8)]),
            Segment(words=[WordTiming("low", 15, 20, 0.1)]),
            Segment(words=[WordTiming("low", 20, 21, 0.1)]),
            Segment(words=[WordTiming("low", 21, 22, 0.1)]),
            Segment(words=[WordTiming("low", 22, 23, 0.1)]),
            Segment(words=[WordTiming("World", 25, 35, 0.8)]),
        ],
        35,
        [
            {"seek": 0, "segments": []},
            {"seek": 25, "segments": [{"start": 0, "end": 10, "text": "World", "word_scores": [0.8]}]},
        ],
        0.8,
        id="low_quality_slice_dropped_cross_over_good_quality",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 10, 0.8)]),
            Segment(words=[WordTiming("low", 15, 20, 0.1)]),
            Segment(words=[WordTiming("low", 20, 21, 0.1)]),
            Segment(words=[WordTiming("low", 21, 22, 0.1)]),
            Segment(words=[WordTiming("low", 22, 23, 0.1)]),
            Segment(words=[WordTiming("World", 35, 45, 0.9)]),
        ],
        45,
        [
            {"seek": 0, "segments": []},
            {"seek": 35, "segments": [{"start": 0, "end": 10, "text": "World", "word_scores": [0.9]}]},
        ],
        0.8,
        id="low_quality_slice_dropped_no_good_quality_in_slice",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 10, 0.8)]),
            Segment(words=[WordTiming("World", 20, 30, 0.9)]),
            Segment(words=[WordTiming("low", 32, 33, 0.1)]),
            Segment(words=[WordTiming("low", 33, 34, 0.1)]),
            Segment(words=[WordTiming("low", 34, 35, 0.1)]),
            Segment(words=[WordTiming("low", 35, 36, 0.1)]),
            Segment(words=[WordTiming("high", 36, 37, 0.9)]),
        ],
        45,
        [
            {
                "seek": 0,
                "segments": [
                    {"start": 5, "end": 10, "text": "Hello", "word_scores": [0.8]},
                    {"start": 20, "end": 30, "text": "World", "word_scores": [0.9]},
                ],
            },
            {"seek": 30, "segments": []},
            {"seek": 36, "segments": [{"start": 0, "end": 1, "text": "high", "word_scores": [0.9]}]},
        ],
        0.8,
        id="low_quality_slice_dropped_not_the_first_slice",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 4, 0.9)]),
            Segment(words=[WordTiming("World", 29, 35, 0.9)]),
        ],
        40,
        [
            {"seek": 0, "segments": [{"start": 2, "end": 4, "text": "Hello", "word_scores": [0.9]}, {"start": 29}]},
            # Segment that crossed over still ends outside the slice, and the only one
            {"seek": 4, "segments": []},
            # so it opens it's own new slice
            {"seek": 29, "segments": [{"start": 0, "end": 6, "text": "World", "word_scores": [0.9]}]},
        ],
        0.5,
        id="twice_crossed_over_push",
    ),
    pytest.param(
        [Segment(words=[WordTiming("Hello", 2, 35, 0.9)])],
        40,
        [],
        0.5,
        id="too_long_for_any_slice_alone",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 4, 0.9)]),
            Segment(words=[WordTiming("World", 29, 60, 0.9)]),
        ],
        62,
        [
            {"seek": 0, "segments": [{"start": 2, "end": 4, "text": "Hello", "word_scores": [0.9]}, {"start": 29}]},
            # Segment that crossed over still ends outside the slice and too long for even a single slice
            # so not created
        ],
        0.5,
        id="too_long_for_any_slice_cross_over",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 35, 0.9)]),
            Segment(words=[WordTiming("World", 36, 56, 0.9)]),
        ],
        60,
        [
            {"seek": 36, "segments": [{"start": 0, "end": 20, "text": "World", "word_scores": [0.9]}]},
            # Segment that crossed over still ends outside the slice and too long for even a single slice
            # so not created
        ],
        0.5,
        id="too_long_for_any_slice_continues_normally",
    ),
    pytest.param(
        [
            Segment(5, 10, "Hello"),
            Segment(15, 20, "World"),
        ],
        15,  # Audio duration shorter than slice_length (30)
        [
            {"seek": 0, "segments": [{"start": 5, "end": 10, "text": "Hello", "word_scores": []}]},
        ],
        0,
        id="audio_duration_shorter_than_slice_length",
    ),
    pytest.param(
        [
            Segment(5, 10, "Hello"),
            Segment(15, 20, "World"),
        ],
        0,  # Zero audio duration
        [],  # Expect empty slices list
        0,
        id="zero_audio_duration",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 4, 0.9), WordTiming(" Again", 5, 8, 0.8)]),
            Segment(words=[WordTiming("Dear", 31, 35, 0.7), WordTiming(" World", 36, 38, 0.6)]),
        ],
        40,
        [
            {
                "seek": 0,
                "segments": [{"start": 2, "end": 8, "text": "Hello Again", "word_scores": [0.9, 0.8]}],
            },
            {"seek": 30, "segments": [{"start": 1, "end": 8, "text": "Dear World", "word_scores": [0.7, 0.6]}]},
        ],
        0.5,
        id="gathers_all_word_scores_of_segments",
    ),
]


@pytest.mark.parametrize("input_segments,audio_duration,expected_slices,per_segment_quality_threshold", test_cases)
@pytest.mark.timeout(2)
def test_generate_slices(input_segments, audio_duration, expected_slices, per_segment_quality_threshold):
    """Test generating slices with parameterized test cases"""
    # Act
    if per_segment_quality_threshold is None:
        result = generate_slices(input_segments, audio_duration, slice_length=30)
    else:
        result = generate_slices(
            input_segments, audio_duration, slice_length=30, per_segment_quality_threshold=per_segment_quality_threshold
        )

    # Assert
    assert len(result) == len(expected_slices), "Should generate expected number of slices"

    if len(result) > 0:
        for slice_idx, (result_slice, expected_slice) in enumerate(zip(result, expected_slices)):
            assert result_slice["seek"] == expected_slice["seek"], f"Slice {slice_idx} seek mismatch"
            assert len(result_slice["segments"]) == len(
                expected_slice["segments"]
            ), f"Slice {slice_idx} should have expected segments"

            for seg_idx, (result_seg, expected_seg) in enumerate(
                zip(result_slice["segments"], expected_slice["segments"])
            ):
                assert result_seg["start"] == expected_seg["start"], f"Segment {seg_idx} start mismatch"
                if "end" in expected_seg:
                    assert result_seg["end"] == expected_seg["end"], f"Segment {seg_idx} end mismatch"
                if "text" in expected_seg:
                    assert result_seg["text"] == expected_seg["text"], f"Segment {seg_idx} text mismatch"
                    assert (
                        result_seg["word_scores"] == expected_seg["word_scores"]
                    ), f"Segment {seg_idx} word scores mismatch"
