import pytest
from stable_whisper.result import Segment, WordTiming

from create_dataset import generate_slices

test_cases = [
    pytest.param(
        [
            Segment(0, 10, "Hello"),
            Segment(15, 20, "World"),
        ],
        30,
        [
            {
                "seek": 0,
                "segments": [{"start": 0, "end": 10, "text": "Hello"}, {"start": 15, "end": 20, "text": "World"}],
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
            {"seek": 0, "segments": [{"start": 0, "end": 10, "text": "Hello"}, {"start": 15}]},
            {"seek": 10, "segments": [{"start": 5, "end": 25, "text": "World"}]},
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
            {"seek": 0, "segments": [{"start": 0, "end": 10, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 50, "segments": [{"start": 0, "end": 20, "text": "World"}]},
        ],
        0,
        id="last_segment_cross_over_single_crossed_over",
    ),
    pytest.param(
        [
            Segment(0, 35, "Hello"),
        ],
        30,
        [
        ],
        0,
        id="segment_end_over_audio_duration",
    ),
    pytest.param(
        [
            Segment(0, 5, "Hello"),
        ],
        10,
        [
            {"seek": 0, "segments": [{"start": 0, "end": 5, "text": "Hello"}]},
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
            {"seek": 30, "segments": [{"start": 5, "end": 15, "text": "Hello"}]},
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
            {"seek": 0, "segments": [{"start": 5, "end": 15, "text": "Hello"}]},
            {"seek": 30, "segments": []},
        ],
        0,
        id="last_slice_empty",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 5, 15, 0.9)]),
            Segment(words=[WordTiming("low", 20, 35, 0.2)]),
        ],
        45,
        [
            {"seek": 0, "segments": [{"start": 5, "end": 15, "text": "Hello"}]},
            {"seek": 30, "segments": []},
        ],
        0.5,
        id="last_low_quality_segment_ignored",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("low", 5, 10, 0.2)]),
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
        ],
        45,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
        ],
        0.5,
        id="first_low_quality_segment_ignored",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
            Segment(words=[WordTiming("low", 31, 35, 0.2)]),
            Segment(words=[WordTiming("Im Pushed", 36, 38, 0.9)]),
            Segment(words=[WordTiming("World", 67, 75, 0.9)]),
        ],
        80,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 35, "segments": [{"start": 1, "end": 3, "text": "Im Pushed"}]},
            {"seek": 65, "segments": [{"start": 2, "end": 10, "text": "World"}]},
        ],
        0.5,
        id="low_quality_start_segment_drops_slice",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
            Segment(words=[WordTiming("Im Gone", 31, 35, 0.9)]),
            Segment(words=[WordTiming("low", 36, 38, 0.2)]),
            Segment(words=[WordTiming("Im Pushed", 39, 42, 0.9)]),
            Segment(words=[WordTiming("World", 69, 75, 0.9)]),
        ],
        80,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 38, "segments": [{"start": 1, "end": 4, "text": "Im Pushed"}]},
            {"seek": 68, "segments": [{"start": 1, "end": 7, "text": "World"}]},
        ],
        0.5,
        id="low_quality_middle_segment_drops_slice",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
            Segment(words=[WordTiming("Im Gone", 31, 35, 0.9)]),
            Segment(words=[WordTiming("low", 36, 38, 0.2)]),
            Segment(words=[WordTiming("World", 67, 75, 0.9)]),
        ],
        80,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 38, "segments": []},
            {"seek": 67, "segments": [{"start": 0, "end": 8, "text": "World"}]},
        ],
        0.5,
        id="low_quality_end_segment_drops_slice",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
            Segment(words=[WordTiming("Im Gone", 31, 35, 0.9)]),
            Segment(words=[WordTiming("low", 36, 38, 0.2)]),
            Segment(words=[WordTiming("World", 61, 69, 0.9)]),
        ],
        80,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 38, "segments": []},
            # The following slice contains a segment that crossed over from the previous slice
            # but was the only segment so it started it's own slice
            {"seek": 61, "segments": [{"start": 0, "end": 8, "text": "World"}]},
        ],
        0.5,
        id="mix_of_drop_and_cross_overs",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 15, 25, 0.9)]),
            Segment(words=[WordTiming("Im Gone", 31, 35, 0.9)]),
            Segment(words=[WordTiming("low", 36, 38, 0.2)]),
            Segment(words=[WordTiming("World", 61, 69, 0.9)]),
            Segment(words=[WordTiming("Crossing", 90, 92, 0.9)]),
        ],
        95,
        [
            {"seek": 0, "segments": [{"start": 15, "end": 25, "text": "Hello"}]},
            {"seek": 30, "segments": []},
            {"seek": 38, "segments": []},
            {"seek": 61, "segments": [{"start": 0, "end": 8, "text": "World"}, {"start": 29}]},
            {"seek": 69, "segments": [{"start": 21, "end": 23, "text": "Crossing"}]},
        ],
        0.5,
        id="mix_of_drop_and_cross_overs_more_complex",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 4, 0.9)]),
            Segment(words=[WordTiming("World", 29, 35, 0.9)]),
        ],
        40,
        [
            {"seek": 0, "segments": [{"start": 2, "end": 4, "text": "Hello"}, {"start": 29}]},
            # Segment that crossed over still ends outside the slice, and the only one
            {"seek": 4, "segments": []},
            # so it opens it's own new slice
            {"seek": 29, "segments": [{"start": 0, "end": 6, "text": "World"}]},
        ],
        0.5,
        id="twice_crossed_over_push",
    ),
    pytest.param(
        [
            Segment(words=[WordTiming("Hello", 2, 35, 0.9)])
        ],
        40,
        [
        ],
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
            {"seek": 0, "segments": [{"start": 2, "end": 4, "text": "Hello"}, {"start": 29}]},
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
            {"seek": 36, "segments": [{"start": 0, "end": 20, "text": "World"}]},
            # Segment that crossed over still ends outside the slice and too long for even a single slice
            # so not created
        ],
        0.5,
        id="too_long_for_any_slice_continues_normally",
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

            for seg_idx, (result_seg, expected_seg) in enumerate(zip(result_slice["segments"], expected_slice["segments"])):
                assert result_seg["start"] == expected_seg["start"], f"Segment {seg_idx} start mismatch"
                if "end" in expected_seg:
                    assert result_seg["end"] == expected_seg["end"], f"Segment {seg_idx} end mismatch"
                if "text" in expected_seg:
                    assert result_seg["text"] == expected_seg["text"], f"Segment {seg_idx} text mismatch"
