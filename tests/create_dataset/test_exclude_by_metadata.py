import json
import tempfile
import unittest
from pathlib import Path

from create_dataset import parse_exclude_filter, get_nested_value, should_exclude_entry


class TestExcludeByMetadata(unittest.TestCase):
    def test_parse_exclude_filter_valid(self):
        """Test parsing valid filter strings."""
        result = parse_exclude_filter("source_id:eq:youtube")
        self.assertEqual(result, ("source_id", "eq", "youtube"))

        result = parse_exclude_filter("metadata.quality_score:eq:0.5")
        self.assertEqual(result, ("metadata.quality_score", "eq", "0.5"))

    def test_parse_exclude_filter_invalid_format(self):
        """Test parsing invalid filter strings."""
        with self.assertRaises(ValueError):
            parse_exclude_filter("invalid_format")

        with self.assertRaises(ValueError):
            parse_exclude_filter("field:invalid_op:value")

    def test_parse_exclude_filter_unsupported_operator(self):
        """Test parsing filter with unsupported operator."""
        with self.assertRaises(ValueError):
            parse_exclude_filter("field:ne:value")

    def test_get_nested_value_simple(self):
        """Test getting simple nested values."""
        obj = {"source_id": "youtube", "quality_score": 0.8}
        self.assertEqual(get_nested_value(obj, "source_id"), "youtube")
        self.assertEqual(get_nested_value(obj, "quality_score"), 0.8)

    def test_get_nested_value_nested(self):
        """Test getting nested values with dot notation."""
        obj = {
            "metadata": {
                "source_id": "youtube",
                "quality_score": 0.8,
                "nested": {"value": "test"}
            }
        }
        self.assertEqual(get_nested_value(obj, "metadata.source_id"), "youtube")
        self.assertEqual(get_nested_value(obj, "metadata.nested.value"), "test")

    def test_get_nested_value_missing_key(self):
        """Test getting values for missing keys."""
        obj = {"source_id": "youtube"}
        self.assertIsNone(get_nested_value(obj, "missing_key"))
        self.assertIsNone(get_nested_value(obj, "metadata.missing"))

    def test_should_exclude_entry_no_filters(self):
        """Test exclusion with no filters."""
        metadata = {"source_id": "youtube"}
        self.assertFalse(should_exclude_entry(metadata, []))

    def test_should_exclude_entry_single_filter_match(self):
        """Test exclusion with single matching filter."""
        metadata = {"source_id": "youtube"}
        filters = [("source_id", "eq", "youtube")]
        self.assertTrue(should_exclude_entry(metadata, filters))

    def test_should_exclude_entry_single_filter_no_match(self):
        """Test exclusion with single non-matching filter."""
        metadata = {"source_id": "youtube"}
        filters = [("source_id", "eq", "vimeo")]
        self.assertFalse(should_exclude_entry(metadata, filters))

    def test_should_exclude_entry_multiple_filters_or_logic(self):
        """Test exclusion with multiple filters using OR logic."""
        metadata = {"source_id": "youtube", "quality_score": 0.5}

        # Should exclude if any filter matches
        filters = [
            ("source_id", "eq", "vimeo"),  # doesn't match
            ("quality_score", "eq", "0.5")  # matches
        ]
        self.assertTrue(should_exclude_entry(metadata, filters))

        # Should not exclude if no filters match
        filters = [
            ("source_id", "eq", "vimeo"),  # doesn't match
            ("quality_score", "eq", "0.8")  # doesn't match
        ]
        self.assertFalse(should_exclude_entry(metadata, filters))

    def test_should_exclude_entry_nested_fields(self):
        """Test exclusion with nested field paths."""
        metadata = {
            "metadata": {
                "source": "youtube",
                "stats": {"views": 1000}
            }
        }

        filters = [("metadata.source", "eq", "youtube")]
        self.assertTrue(should_exclude_entry(metadata, filters))

        filters = [("metadata.stats.views", "eq", "1000")]
        self.assertTrue(should_exclude_entry(metadata, filters))

        filters = [("metadata.stats.views", "eq", "2000")]
        self.assertFalse(should_exclude_entry(metadata, filters))


if __name__ == "__main__":
    unittest.main()
