"""Tests for the journal_partition utility."""

from __future__ import annotations

from academic_assistant.utils.journal_partition import lookup_partition, JOURNAL_PARTITION_MAP


class TestLookupPartition:
    def test_known_journal_returns_partition(self):
        assert lookup_partition("Nature") == "SCI Q1"
        assert lookup_partition("IEEE Access") == "EI"
        assert lookup_partition("计算机学报") == "CSCD"

    def test_case_insensitive_match(self):
        assert lookup_partition("NATURE") == "SCI Q1"
        assert lookup_partition("nature") == "SCI Q1"
        assert lookup_partition("Nature Communications") == "SCI Q1"

    def test_leading_trailing_whitespace_ignored(self):
        assert lookup_partition("  Nature  ") == "SCI Q1"

    def test_unknown_journal_returns_none(self):
        assert lookup_partition("Some Completely Unknown Journal XYZ") is None

    def test_none_input_returns_none(self):
        assert lookup_partition(None) is None

    def test_empty_string_returns_none(self):
        assert lookup_partition("") is None

    def test_map_contains_q1_q4_ei_cscd(self):
        partitions = set(JOURNAL_PARTITION_MAP.values())
        assert "SCI Q1" in partitions
        assert "SCI Q2" in partitions
        assert "EI" in partitions
        assert "CSCD" in partitions
        assert "北大核心" in partitions
