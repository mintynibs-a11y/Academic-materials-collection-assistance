"""Tests for the AcademicAssistant orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from academic_assistant.assistant import AcademicAssistant
from academic_assistant.models.paper import Paper, ResearchReport, SearchResult, Source


def _make_result(source: Source, titles: list[str], error: str | None = None) -> SearchResult:
    papers = [Paper(title=t, source=source) for t in titles]
    return SearchResult(query="test", source=source, papers=papers, error=error)


class TestAcademicAssistant:
    def test_aggregate_deduplicates_by_title(self):
        results = [
            _make_result(Source.IEEE, ["Paper A", "Paper B"]),
            _make_result(Source.WEB, ["Paper B", "Paper C"]),  # "Paper B" is duplicate
        ]
        papers = AcademicAssistant._aggregate(results)
        titles = [p.title for p in papers]
        assert titles.count("Paper B") == 1
        assert len(papers) == 3

    def test_aggregate_skips_error_sources(self):
        results = [
            _make_result(Source.IEEE, ["Paper A"]),
            _make_result(Source.WEB, [], error="Network error"),
        ]
        papers = AcademicAssistant._aggregate(results)
        assert len(papers) == 1

    def test_aggregate_case_insensitive_dedup(self):
        results = [
            _make_result(Source.IEEE, ["Deep Learning in Healthcare"]),
            _make_result(Source.WEB, ["deep learning in healthcare"]),  # same, different case
        ]
        papers = AcademicAssistant._aggregate(results)
        assert len(papers) == 1

    async def test_research_returns_report(self):
        assistant = AcademicAssistant(max_results_per_source=2)

        mock_papers = [
            Paper(title="Paper 1", year=2023, source=Source.IEEE),
            Paper(title="Paper 2", year=2022, source=Source.WEB),
        ]
        mock_search_result = SearchResult(
            query="AI", source=Source.IEEE, papers=mock_papers
        )

        mock_report = ResearchReport(
            query="AI research",
            sources_searched=[Source.IEEE],
            total_papers_found=2,
            ranked_papers=[],
            summary="AI is advancing rapidly.",
        )

        with (
            patch.object(assistant, "_search_all", new=AsyncMock(return_value=[mock_search_result])),
            patch.object(assistant._ranker, "rank_and_summarize", new=AsyncMock(return_value=mock_report)),
        ):
            report = await assistant.research("AI research")

        assert report.query == "AI research"
        assert report.summary == "AI is advancing rapidly."

    def test_print_report_does_not_raise(self, capsys):
        paper = Paper(title="Test", authors=["Alice"], year=2024, source=Source.IEEE)
        from academic_assistant.models.paper import RankedPaper

        ranked = RankedPaper(paper=paper, rank=1, relevance_score=0.9, reason="Very relevant")
        report = ResearchReport(
            query="test query",
            sources_searched=[Source.IEEE],
            total_papers_found=1,
            ranked_papers=[ranked],
            summary="This is a test summary.",
            key_themes=["AI", "ML"],
            research_gaps=["Interpretability"],
        )
        AcademicAssistant.print_report(report)
        captured = capsys.readouterr()
        assert "test query" in captured.out
        assert "Test" in captured.out
        assert "AI" in captured.out
        assert "Interpretability" in captured.out
