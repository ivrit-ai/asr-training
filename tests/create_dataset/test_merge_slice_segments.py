import pytest
from create_dataset import merge_slice_segments


def test_empty_slices():
    """Test with empty slices list."""
    assert merge_slice_segments([]) == []


def test_slice_with_no_segments():
    """Test with a slice that has no segments."""
    slices = [{"seek": 0, "segments": []}]
    assert merge_slice_segments(slices) == slices


def test_slice_with_one_segment():
    """Test with a slice that has only one segment."""
    slices = [{"seek": 0, "segments": [{"start": 0, "end": 5, "text": "Hello"}]}]
    assert merge_slice_segments(slices) == slices


def test_no_merge_gap_too_large():
    """Test when segments have a gap larger than the threshold."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.5, "end": 10, "text": "world"},
            ],
        }
    ]
    # Gap is 0.5s, which is > default threshold of 0.3s
    assert merge_slice_segments(slices) == slices


def test_merge_segments_within_threshold():
    """Test merging segments with a gap smaller than the threshold."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.2, "end": 10, "text": "world"},
            ],
        }
    ]
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 10, "text": "Hello world"},
            ],
        }
    ]
    # Gap is 0.2s, which is < default threshold of 0.3s
    assert merge_slice_segments(slices) == expected


def test_merge_multiple_segments():
    """Test merging multiple segments in sequence."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.1, "end": 10, "text": "world "},
                {"start": 10.2, "end": 15, "text": "today"},
            ],
        }
    ]
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 15, "text": "Hello world today"},
            ],
        }
    ]
    assert merge_slice_segments(slices) == expected


def test_merge_some_segments_not_others():
    """Test when some segments can be merged but others cannot."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.1, "end": 10, "text": "world "},
                {"start": 10.5, "end": 15, "text": "today"},
            ],
        }
    ]
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 10, "text": "Hello world "},
                {"start": 10.5, "end": 15, "text": "today"},
            ],
        }
    ]
    assert merge_slice_segments(slices) == expected


def test_multiple_slices():
    """Test with multiple slices."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.1, "end": 10, "text": "world"},
            ],
        },
        {
            "seek": 30,
            "segments": [
                {"start": 0, "end": 5, "text": "Another "},
                {"start": 5.5, "end": 10, "text": "segment"},
            ],
        },
    ]
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 10, "text": "Hello world"},
            ],
        },
        {
            "seek": 30,
            "segments": [
                {"start": 0, "end": 5, "text": "Another "},
                {"start": 5.5, "end": 10, "text": "segment"},
            ],
        },
    ]
    assert merge_slice_segments(slices) == expected


def test_custom_thresholds():
    """Test with custom threshold values."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.5, "end": 10, "text": "world "},
                {"start": 15, "end": 20, "text": "today"},
            ],
        }
    ]
    # Using a larger gap threshold (1.0s) to merge the first two segments
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 10, "text": "Hello world "},
                {"start": 15, "end": 20, "text": "today"},
            ],
        }
    ]
    assert merge_slice_segments(slices, merge_below_gap_threshold=1.0) == expected


def test_incomplete_segments():
    """Test with segments missing required keys."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "Hello "},
                {"start": 5.1},  # Missing end and text
            ],
        }
    ]
    # Should not merge due to missing keys
    assert merge_slice_segments(slices) == slices


def test_complex_scenario():
    """Test a more complex scenario with multiple merges and non-merges."""
    slices = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 5, "text": "First "},
                {"start": 5.2, "end": 10, "text": "second "},
                {"start": 10.5, "end": 15, "text": "third "},
                {"start": 15.1, "end": 20, "text": "fourth "},
                {"start": 20.2, "end": 26, "text": "fifth "},
                {"start": 26.1},
            ],
        }
    ]
    # Our implementation should merge:
    # 1. "First " with "second " (gap of 0.2s < threshold)
    # 2. "third " with "fourth " (gap of 0.1s < threshold)
    # 3. "fourth fifth " with "fifth " (gap of 0.2s < threshold)
    # But not merge with "sixth" because it starts above the threshold (26.1 > 26.0)
    expected = [
        {
            "seek": 0,
            "segments": [
                {"start": 0, "end": 10, "text": "First second "},
                {"start": 10.5, "end": 26, "text": "third fourth fifth "},
                {"start": 26.1},
            ],
        }
    ]
    assert merge_slice_segments(slices) == expected
