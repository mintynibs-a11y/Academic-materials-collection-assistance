"""Tests for the PaperRanker processor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from academic_assistant.models.paper import Paper, Source
from academic_assistant.processors.ranker import PaperRanker


SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A."],
        "year": 2017,
        "abstract": "We propose the Transformer architecture based entirely on attention mechanisms.",
        "citations": 90000,
        "source": "web",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J."],
        "year": 2018,
        "abstract": "We introduce BERT, a new language representation model.",
        "citations": 50000,
        "source": "ieee",
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
        assert report.ranked_papers[0].relevance_score == pytest.approx(0.98)

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

    def test_dict_to_paper_valid(self):
        data = {"title": "Test", "year": 2020, "source": "ieee"}
        paper = PaperRanker._dict_to_paper(data)
        assert paper.title == "Test"
        assert paper.year == 2020

    def test_dict_to_paper_invalid_graceful(self):
        """Should not raise even with malformed data."""
        data = {"title": "Test", "year": "not-a-year", "relevance_score": 99.0}
        paper = PaperRanker._dict_to_paper(data)
        assert paper.title == "Test"
        assert paper.source == Source.UNKNOWN
