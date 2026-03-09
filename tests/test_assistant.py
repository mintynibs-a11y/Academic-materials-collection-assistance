"""Tests for the AcademicAssistant orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from academic_assistant.assistant import AcademicAssistant
from academic_assistant.models.paper import NewsItem, Paper, ResearchReport, SearchResult, Source


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

    def test_aggregate_annotates_journal_partition(self):
        """Papers with known journals should get partition annotated by _aggregate."""
        papers_in = [Paper(title="P1", journal="Nature", source=Source.WEB_OF_SCIENCE)]
        result = SearchResult(query="q", source=Source.WEB_OF_SCIENCE, papers=papers_in)
        papers = AcademicAssistant._aggregate([result])
        assert papers[0].journal_partition == "SCI Q1"

    def test_aggregate_unknown_journal_leaves_none(self):
        papers_in = [Paper(title="P2", journal="Obscure Journal XYZ", source=Source.WEB)]
        result = SearchResult(query="q", source=Source.WEB, papers=papers_in)
        papers = AcademicAssistant._aggregate([result])
        assert papers[0].journal_partition is None

    async def test_research_with_keywords(self):
        """research() should accept keywords list and build query correctly."""
        assistant = AcademicAssistant(max_results_per_source=2)

        mock_papers = [Paper(title="Paper 1", year=2023, source=Source.IEEE)]
        mock_search_result = SearchResult(
            query="deep learning AND protein folding",
            source=Source.IEEE,
            papers=mock_papers,
        )
        mock_report = ResearchReport(
            query="deep learning; protein folding",
            sources_searched=[Source.IEEE],
            total_papers_found=1,
            ranked_papers=[],
            summary="Interesting findings.",
        )

        search_all_mock = AsyncMock(return_value=[mock_search_result])
        with (
            patch.object(assistant, "_search_all", search_all_mock),
            patch.object(assistant._ranker, "rank_and_summarize", new=AsyncMock(return_value=mock_report)),
            patch.object(assistant, "_fetch_news", new=AsyncMock(return_value=[])),
            patch.object(assistant._ranker, "summarize_news", new=AsyncMock(return_value="")),
        ):
            report = await assistant.research(
                keywords=["deep learning", "protein folding"], top_n=5
            )

        assert report.query == "deep learning; protein folding"
        # The _search_all should have been called with "deep learning AND protein folding"
        search_all_mock.assert_called_once_with("deep learning AND protein folding")

    async def test_research_with_query_string(self):
        """research() should still work with a plain query string."""
        assistant = AcademicAssistant(max_results_per_source=2)

        mock_search_result = SearchResult(
            query="AI", source=Source.IEEE,
            papers=[Paper(title="Paper 1", source=Source.IEEE)]
        )
        mock_report = ResearchReport(
            query="AI research",
            sources_searched=[Source.IEEE],
            total_papers_found=1,
            ranked_papers=[],
            summary="AI is advancing rapidly.",
        )

        with (
            patch.object(assistant, "_search_all", new=AsyncMock(return_value=[mock_search_result])),
            patch.object(assistant._ranker, "rank_and_summarize", new=AsyncMock(return_value=mock_report)),
        ):
            report = await assistant.research(query="AI research", include_news=False)

        assert report.query == "AI research"

    async def test_research_includes_news(self):
        """research() with keywords should populate news_items in the report."""
        assistant = AcademicAssistant(max_results_per_source=2)

        mock_news = [NewsItem(title="AI Breakthrough", url="https://example.com")]
        mock_report = ResearchReport(
            query="transformer",
            sources_searched=[],
            total_papers_found=0,
            ranked_papers=[],
            summary="Summary.",
        )

        with (
            patch.object(assistant, "_search_all", new=AsyncMock(return_value=[])),
            patch.object(assistant._ranker, "rank_and_summarize", new=AsyncMock(return_value=mock_report)),
            patch.object(assistant, "_fetch_news", new=AsyncMock(return_value=mock_news)),
            patch.object(assistant._ranker, "summarize_news", new=AsyncMock(return_value="News summary.")),
        ):
            report = await assistant.research(keywords=["transformer"], top_n=5)

        assert len(report.news_items) == 1
        assert report.news_items[0].title == "AI Breakthrough"
        assert report.news_summary == "News summary."

    def test_print_report_does_not_raise(self, capsys):
        paper = Paper(title="Test", authors=["Alice"], year=2024, source=Source.IEEE,
                      journal="Nature", journal_partition="SCI Q1",
                      abstract="This paper studies AI applications in healthcare.")
        from academic_assistant.models.paper import RankedPaper

        ranked = RankedPaper(paper=paper, rank=1, relevance_score=0.9, reason="Very relevant")
        report = ResearchReport(
            query="test query",
            keywords=["test", "query"],
            sources_searched=[Source.IEEE],
            total_papers_found=1,
            ranked_papers=[ranked],
            summary="This is a test summary.",
            key_themes=["AI", "ML"],
            research_gaps=["Interpretability"],
            news_items=[NewsItem(title="News item", url="https://example.com", snippet="Short snippet about AI research today.")],
            news_summary="Recent AI news summary.",
        )
        AcademicAssistant.print_report(report)
        captured = capsys.readouterr()
        assert "test query" in captured.out
        assert "Test" in captured.out
        assert "Nature" in captured.out
        assert "SCI Q1" in captured.out
        assert "AI" in captured.out
        assert "Interpretability" in captured.out
        assert "News item" in captured.out
        assert "Recent AI news summary" in captured.out
