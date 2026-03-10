"""Tests for search adapters."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from academic_assistant.models.paper import Paper, Source
from academic_assistant.searchers.web_of_science import WebOfScienceSearcher
from academic_assistant.searchers.ieee import IEEESearcher
from academic_assistant.searchers.web_search import WebSearcher
from academic_assistant.searchers.cnki import CNKISearcher


class TestWebOfScienceSearcher:
    async def test_no_api_key_returns_error(self):
        searcher = WebOfScienceSearcher(api_key="")
        result = await searcher.search("machine learning")
        assert not result.success
        assert "WOS_API_KEY" in result.error

    async def test_http_error_returns_error(self):
        searcher = WebOfScienceSearcher(api_key="fake-key")
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response
            )
            result = await searcher.search("test")
        assert not result.success
        assert "401" in result.error

    async def test_parse_response(self):
        searcher = WebOfScienceSearcher(api_key="fake-key")
        data = {
            "hits": [
                {
                    "title": "Test Paper",
                    "names": {"authors": [{"displayName": "Alice Smith"}, {"displayName": "Bob Jones"}]},
                    "source": {"sourceTitle": "Nature", "publishDate": "2023-01-15"},
                    "identifiers": [{"type": "doi", "value": "10.1234/test"}],
                    "citations": [{"count": 42}],
                    "keywords": {"authorKeywords": [{"value": "ML"}, {"value": "AI"}]},
                    "uid": "WOS:000123",
                }
            ],
            "metadata": {"total": 100},
        }
        papers = searcher._parse_response(data)
        assert len(papers) == 1
        p = papers[0]
        assert p.title == "Test Paper"
        assert p.authors == ["Alice Smith", "Bob Jones"]
        assert p.journal == "Nature"
        assert p.year == 2023
        assert p.doi == "10.1234/test"
        assert p.citations == 42
        assert "ML" in p.keywords
        assert p.source == Source.WEB_OF_SCIENCE


class TestIEEESearcher:
    async def test_no_api_key_returns_error(self):
        searcher = IEEESearcher(api_key="")
        result = await searcher.search("deep learning")
        assert not result.success
        assert "IEEE_API_KEY" in result.error

    async def test_parse_response(self):
        searcher = IEEESearcher(api_key="fake-key")
        data = {
            "total_records": 50,
            "articles": [
                {
                    "title": "Deep Neural Networks",
                    "authors": {"authors": [{"full_name": "Jane Doe"}]},
                    "publication_year": 2022,
                    "publication_title": "IEEE Transactions on Neural Networks",
                    "doi": "10.1109/TNN.2022.001",
                    "article_number": "9876543",
                    "abstract": "A study of deep learning.",
                    "citing_paper_count": 100,
                    "index_terms": {
                        "author_terms": {"terms": ["neural networks", "deep learning"]}
                    },
                }
            ],
        }
        papers = searcher._parse_response(data)
        assert len(papers) == 1
        p = papers[0]
        assert p.title == "Deep Neural Networks"
        assert p.authors == ["Jane Doe"]
        assert p.year == 2022
        assert p.journal == "IEEE Transactions on Neural Networks"
        assert p.doi == "10.1109/TNN.2022.001"
        assert p.url == "https://ieeexplore.ieee.org/document/9876543"
        assert p.citations == 100
        assert "neural networks" in p.keywords


class TestWebSearcher:
    async def test_serper_search(self):
        searcher = WebSearcher(api_key="fake-serper-key")
        searcher._serper_key = "fake-serper-key"
        mock_data = {
            "organic": [
                {
                    "title": "Paper on AI",
                    "link": "https://example.com/paper",
                    "snippet": "A 2023 study on AI applications...",
                }
            ]
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_data)
            mock_post.return_value = mock_resp
            result = await searcher.search("AI research", max_results=5)

        assert result.success
        assert len(result.papers) == 1
        assert result.papers[0].title == "Paper on AI"
        assert result.papers[0].year == 2023

    async def test_parse_serper_extracts_year(self):
        searcher = WebSearcher()
        data = {
            "organic": [
                {"title": "Old paper", "snippet": "Published in 1998"},
                {"title": "New paper", "snippet": "A 2024 breakthrough study"},
            ]
        }
        papers = searcher._parse_serper(data)
        assert len(papers) == 2
        assert papers[0].year == 1998
        assert papers[1].year == 2024


class TestCNKISearcher:
    async def test_request_failure_returns_error(self):
        searcher = CNKISearcher()
        import httpx

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = await searcher.search("人工智能")
        assert not result.success
        assert "CNKI request failed" in result.error

    def test_parse_html_empty(self):
        searcher = CNKISearcher()
        papers = searcher._parse_html("<html><body></body></html>", 10)
        assert papers == []


class TestWebNewsSearcher:

    async def test_serper_news_returns_items(self):
        from academic_assistant.searchers.web_news import WebNewsSearcher

        searcher = WebNewsSearcher()
        searcher._serper_key = "fake-key"
        searcher._brave_key = ""

        mock_data = {
            "news": [
                {
                    "title": "AI in Medicine",
                    "link": "https://example.com/news",
                    "snippet": "Researchers apply AI to diagnose diseases.",
                    "source": "TechNews",
                    "date": "2024-03-01",
                }
            ]
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_data)
            mock_post.return_value = mock_resp
            items = await searcher.search_news(["AI", "medicine"], max_results=5)

        assert len(items) == 1
        assert items[0].title == "AI in Medicine"
        assert items[0].url == "https://example.com/news"
        assert items[0].source_name == "TechNews"

    async def test_network_failure_returns_empty_list(self):
        from academic_assistant.searchers.web_news import WebNewsSearcher
        import httpx

        searcher = WebNewsSearcher()
        searcher._serper_key = ""
        searcher._brave_key = ""

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            items = await searcher.search_news(["AI"], max_results=5)

        assert items == []

    async def test_empty_keywords_returns_items_or_empty(self):
        """search_news with empty keywords should not raise."""
        from academic_assistant.searchers.web_news import WebNewsSearcher
        import httpx

        searcher = WebNewsSearcher()
        searcher._serper_key = ""
        searcher._brave_key = ""

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("No network")
            items = await searcher.search_news([], max_results=5)

        assert isinstance(items, list)

