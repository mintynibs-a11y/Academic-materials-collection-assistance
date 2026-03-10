"""Tests for the PaperRanker processor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from academic_assistant.models.paper import NewsItem, Paper, Source
from academic_assistant.processors.ranker import PaperRanker


SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A."],
        "year": 2017,
        "abstract": "We propose the Transformer architecture based entirely on attention mechanisms.",
        "citations": 90000,
        "source": "web",
        "journal_partition": "SCI Q1",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J."],
        "year": 2018,
        "abstract": "We introduce BERT, a new language representation model.",
        "citations": 50000,
        "source": "ieee",
        "journal_partition": None,
    },
]

SAMPLE_LLM_RESPONSE = {
    "ranked_papers": [
        {"index": 0, "relevance_score": 0.98, "reason": "Foundational transformer paper."},
        {"index": 1, "relevance_score": 0.95, "reason": "Key BERT paper."},
    ],
    "summary": "Transformer-based models have revolutionised NLP.",
    "key_themes": ["attention mechanisms", "pre-training", "language models"],
    "research_gaps": ["efficiency", "multilingual support"],
}

SAMPLE_NEWS_ITEMS = [
    NewsItem(
        title="New transformer model breaks records",
        url="https://example.com/news1",
        snippet="Researchers achieve state-of-the-art results.",
    ),
    NewsItem(
        title="LLM applications in healthcare",
        url="https://example.com/news2",
        snippet="Hospitals adopt AI for diagnosis.",
    ),
]

SAMPLE_NEWS_LLM_RESPONSE = {
    "summary": "Recent advances show transformers dominating NLP and medical AI.",
    "highlights": [
        {
            "title": "New transformer model breaks records",
            "url": "https://example.com/news1",
            "description": "Achieves SOTA on multiple benchmarks.",
        }
    ],
}


class TestPaperRanker:
    async def test_rank_with_openai(self):
        ranker = PaperRanker()
        ranker._provider = "openai"

        with patch.object(ranker, "_call_openai", new=AsyncMock(return_value=SAMPLE_LLM_RESPONSE)):
            report = await ranker.rank_and_summarize("transformer models", SAMPLE_PAPERS, top_n=2)

        assert report.total_papers_found == 2
        assert len(report.ranked_papers) == 2
        assert report.ranked_papers[0].rank == 1
        assert report.ranked_papers[0].relevance_score == pytest.approx(0.98)
        assert "Transformer" in report.summary
        assert "attention mechanisms" in report.key_themes

    async def test_rank_includes_journal_partition_in_compact(self):
        """journal_partition must be included in the compact dict sent to LLM."""
        ranker = PaperRanker()
        ranker._provider = "openai"

        captured_compact: list = []

        async def _fake_call_llm(query, compact_papers, top_n):
            captured_compact.extend(compact_papers)
            return SAMPLE_LLM_RESPONSE

        with patch.object(ranker, "_call_llm", new=_fake_call_llm):
            await ranker.rank_and_summarize("transformer models", SAMPLE_PAPERS, top_n=2)

        assert "journal_partition" in captured_compact[0]
        assert captured_compact[0]["journal_partition"] == "SCI Q1"

    async def test_rank_with_anthropic(self):
        ranker = PaperRanker()
        ranker._provider = "anthropic"

        with patch.object(ranker, "_call_anthropic", new=AsyncMock(return_value=SAMPLE_LLM_RESPONSE)):
            report = await ranker.rank_and_summarize("transformer models", SAMPLE_PAPERS, top_n=2)

        assert report.total_papers_found == 2
        assert len(report.ranked_papers) == 2

    async def test_rank_with_deepseek(self):
        ranker = PaperRanker()
        ranker._provider = "deepseek"

        with patch.object(ranker, "_call_deepseek", new=AsyncMock(return_value=SAMPLE_LLM_RESPONSE)):
            report = await ranker.rank_and_summarize("transformer models", SAMPLE_PAPERS, top_n=2)

        assert report.total_papers_found == 2
        assert len(report.ranked_papers) == 2
        assert report.ranked_papers[0].rank == 1
        assert report.ranked_papers[0].relevance_score == pytest.approx(0.98)
        assert report.ranked_papers[0].paper.title == "Attention Is All You Need"
        assert report.ranked_papers[1].relevance_score == pytest.approx(0.95)

    async def test_empty_papers_returns_empty_report(self):
        ranker = PaperRanker()
        report = await ranker.rank_and_summarize("test", [], top_n=5)
        assert report.total_papers_found == 0
        assert report.ranked_papers == []
        assert "No papers" in report.summary

    async def test_llm_failure_returns_empty_ranking(self):
        ranker = PaperRanker()
        ranker._provider = "openai"

        with patch.object(ranker, "_call_openai", new=AsyncMock(return_value={})):
            report = await ranker.rank_and_summarize("test", SAMPLE_PAPERS, top_n=5)

        # No ranked papers if LLM returns empty
        assert report.total_papers_found == 2
        assert report.ranked_papers == []

    async def test_summarize_news_returns_summary(self):
        ranker = PaperRanker()
        ranker._provider = "openai"

        with patch.object(
            ranker,
            "_call_openai_raw",
            new=AsyncMock(return_value=SAMPLE_NEWS_LLM_RESPONSE),
        ):
            summary = await ranker.summarize_news(
                ["transformer", "NLP"], SAMPLE_NEWS_ITEMS
            )

        assert "transformer" in summary.lower() or len(summary) > 0

    async def test_summarize_news_empty_items_returns_empty_string(self):
        ranker = PaperRanker()
        summary = await ranker.summarize_news(["AI"], [])
        assert summary == ""

    def test_dict_to_paper_valid(self):
        data = {"title": "Test", "year": 2020, "source": "ieee"}
        paper = PaperRanker._dict_to_paper(data)
        assert paper.title == "Test"
        assert paper.year == 2020

    def test_dict_to_paper_with_journal_partition(self):
        data = {"title": "Test", "journal": "Nature", "journal_partition": "SCI Q1", "source": "web_of_science"}
        paper = PaperRanker._dict_to_paper(data)
        assert paper.journal_partition == "SCI Q1"

    def test_dict_to_paper_invalid_graceful(self):
        """Should not raise even with malformed data."""
        data = {"title": "Test", "year": "not-a-year", "relevance_score": 99.0}
        paper = PaperRanker._dict_to_paper(data)
        assert paper.title == "Test"
        assert paper.source == Source.UNKNOWN
