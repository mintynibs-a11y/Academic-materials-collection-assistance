"""
General web search adapter.

Supports two backends (first available is used):
  1. Serper (Google Search API) – set SERPER_API_KEY
  2. Brave Search API            – set BRAVE_API_KEY

Falls back to a DuckDuckGo HTML scrape when neither key is set.
"""

from __future__ import annotations

import logging
import os
import re

import httpx
from bs4 import BeautifulSoup

from academic_assistant.models.paper import Paper, SearchResult, Source
from academic_assistant.searchers.base import BaseSearcher

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = os.getenv(
    "ACADEMIC_ASSISTANT_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
)


class WebSearcher(BaseSearcher):
    """General web search adapter for finding academic content on the open web."""

    source = Source.WEB

    def __init__(self, api_key: str = "") -> None:
        super().__init__(api_key)
        self._serper_key = os.getenv("SERPER_API_KEY", api_key)
        self._brave_key = os.getenv("BRAVE_API_KEY", "")

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        if self._serper_key:
            return await self._serper_search(query, max_results)
        if self._brave_key:
            return await self._brave_search(query, max_results)
        return await self._ddg_search(query, max_results)

    # ------------------------------------------------------------------
    # Backend: Serper
    # ------------------------------------------------------------------

    async def _serper_search(self, query: str, max_results: int) -> SearchResult:
        payload = {"q": query + " academic paper", "num": min(max_results, 10)}
        headers = {"X-API-KEY": self._serper_key, "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://google.serper.dev/search", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return self._error_result(query, f"Serper error: {exc}")

        papers = self._parse_serper(data)
        return SearchResult(query=query, source=self.source, papers=papers)

    def _parse_serper(self, data: dict) -> list[Paper]:
        papers: list[Paper] = []
        for item in data.get("organic", []):
            title = item.get("title", "")
            if not title:
                continue
            year_match = re.search(r"\b(19|20)\d{2}\b", item.get("snippet", ""))
            year = int(year_match.group()) if year_match else None
            papers.append(
                Paper(
                    title=title,
                    abstract=item.get("snippet"),
                    url=item.get("link"),
                    year=year,
                    source=self.source,
                )
            )
        return papers

    # ------------------------------------------------------------------
    # Backend: Brave Search
    # ------------------------------------------------------------------

    async def _brave_search(self, query: str, max_results: int) -> SearchResult:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_key,
        }
        params = {"q": query + " academic paper", "count": min(max_results, 20)}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return self._error_result(query, f"Brave Search error: {exc}")

        papers: list[Paper] = []
        for result in data.get("web", {}).get("results", []):
            title = result.get("title", "")
            if not title:
                continue
            year_match = re.search(r"\b(19|20)\d{2}\b", result.get("description", ""))
            year = int(year_match.group()) if year_match else None
            papers.append(
                Paper(
                    title=title,
                    abstract=result.get("description"),
                    url=result.get("url"),
                    year=year,
                    source=self.source,
                )
            )
        return SearchResult(query=query, source=self.source, papers=papers)

    # ------------------------------------------------------------------
    # Backend: DuckDuckGo HTML fallback (no API key required)
    # ------------------------------------------------------------------

    async def _ddg_search(self, query: str, max_results: int) -> SearchResult:
        headers = {"User-Agent": _DEFAULT_USER_AGENT}
        params = {"q": query + " academic paper filetype:pdf OR site:scholar.google.com"}
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/", params=params, headers=headers
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            return self._error_result(query, f"DuckDuckGo fallback error: {exc}")

        soup = BeautifulSoup(html, "lxml")
        papers: list[Paper] = []
        for result in soup.select(".result__body")[:max_results]:
            title_tag = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            url = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
            year_match = re.search(r"\b(19|20)\d{2}\b", snippet or "")
            year = int(year_match.group()) if year_match else None
            papers.append(
                Paper(
                    title=title,
                    abstract=snippet,
                    url=url or None,
                    year=year,
                    source=self.source,
                )
            )
        return SearchResult(query=query, source=self.source, papers=papers)
