"""Tests for data models."""

from __future__ import annotations

import pytest
from academic_assistant.models.paper import (
    Paper,
    RankedPaper,
    ResearchReport,
    SearchResult,
    Source,
)


class TestPaper:
    def test_minimal_paper(self):
        paper = Paper(title="Test Paper")
        assert paper.title == "Test Paper"
        assert paper.authors == []
        assert paper.source == Source.UNKNOWN

    def test_formatted_authors_empty(self):
        paper = Paper(title="T")
        assert paper.formatted_authors == "Unknown authors"

    def test_formatted_authors_single(self):
        paper = Paper(title="T", authors=["Alice Smith"])
        assert paper.formatted_authors == "Alice Smith"

    def test_formatted_authors_three(self):
        paper = Paper(title="T", authors=["A", "B", "C"])
        assert paper.formatted_authors == "A, B, C"

    def test_formatted_authors_et_al(self):
        paper = Paper(title="T", authors=["A", "B", "C", "D"])
        assert paper.formatted_authors == "A et al."

    def test_short_summary(self):
        paper = Paper(title="My Paper", authors=["Alice"], year=2023, citations=42, source=Source.IEEE)
        summary = paper.short_summary()
        assert "My Paper" in summary
        assert "2023" in summary
        assert "cited 42" in summary

    def test_relevance_score_validation(self):
        paper = Paper(title="T", relevance_score=0.75)
        assert paper.relevance_score == 0.75

        with pytest.raises(Exception):
            Paper(title="T", relevance_score=1.5)  # out of range

    def test_year_validation(self):
        with pytest.raises(Exception):
            Paper(title="T", year=999)  # too old (before year 1000)

    def test_source_enum_serialization(self):
        paper = Paper(title="T", source=Source.IEEE)
        data = paper.model_dump()
        assert data["source"] == "ieee"


class TestSearchResult:
    def test_success_result(self):
        papers = [Paper(title="P1"), Paper(title="P2")]
        result = SearchResult(query="q", source=Source.WEB, papers=papers)
        assert result.success is True
        assert len(result.papers) == 2

    def test_error_result(self):
        result = SearchResult(query="q", source=Source.WEB, error="Connection timeout")
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.papers == []


class TestResearchReport:
    def test_empty_report(self):
        report = ResearchReport(
            query="test",
            sources_searched=[],
            total_papers_found=0,
            ranked_papers=[],
            summary="Nothing found.",
        )
        assert report.total_papers_found == 0
        assert report.key_themes == []
        assert report.research_gaps == []

    def test_full_report(self):
        paper = Paper(title="Great Paper", year=2024, source=Source.IEEE)
        ranked = RankedPaper(paper=paper, rank=1, relevance_score=0.95, reason="Highly relevant")
        report = ResearchReport(
            query="deep learning",
            sources_searched=[Source.IEEE, Source.WEB],
            total_papers_found=5,
            ranked_papers=[ranked],
            summary="Deep learning is advancing.",
            key_themes=["transformers", "efficiency"],
            research_gaps=["few-shot learning"],
        )
        assert len(report.ranked_papers) == 1
        assert report.ranked_papers[0].rank == 1
        assert len(report.key_themes) == 2
