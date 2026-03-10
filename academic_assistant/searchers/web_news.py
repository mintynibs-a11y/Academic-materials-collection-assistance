"""
Web research-news scraper.

Searches the open web for recent scientific news and updates related to
the given keywords.  Backends (in priority order):
  1. Serper  (set SERPER_API_KEY)
  2. Brave Search  (set BRAVE_API_KEY)
  3. DuckDuckGo HTML fallback  (no key required)

The results are returned as :class:`~academic_assistant.models.paper.NewsItem`
objects rather than ``Paper`` objects because they are general-purpose news
articles, not peer-reviewed publications.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from academic_assistant.models.paper import NewsItem

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = os.getenv(
    "ACADEMIC_ASSISTANT_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
)

# Appended to every news query to bias results toward research content
_NEWS_SUFFIX = "科研进展 最新研究 research news"


class WebNewsSearcher:
    """Scrapes research-news articles for a list of keywords."""

    def __init__(self) -> None:
        self._serper_key = os.getenv("SERPER_API_KEY", "")
        self._brave_key = os.getenv("BRAVE_API_KEY", "")

    async def search_news(
        self, keywords: list[str], max_results: int = 10
    ) -> list[NewsItem]:
        """
        Return recent research-news articles related to *keywords*.

        Parameters
        ----------
        keywords:
            List of search keywords.
        max_results:
            Maximum number of news items to return.

        Returns
        -------
        list[NewsItem]
            Empty list on failure (errors are logged, not raised).
        """
        query = " ".join(keywords) + " " + _NEWS_SUFFIX
        try:
            if self._serper_key:
                return await self._serper_news(query, max_results)
            if self._brave_key:
                return await self._brave_news(query, max_results)
            return await self._ddg_news(query, max_results)
        except Exception as exc:
            logger.error("WebNewsSearcher failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Backend: Serper
    # ------------------------------------------------------------------

    async def _serper_news(self, query: str, max_results: int) -> list[NewsItem]:
        payload = {"q": query, "num": min(max_results, 10), "type": "news"}
        headers = {"X-API-KEY": self._serper_key, "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://google.serper.dev/news", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Serper news search failed: %s", exc)
            return await self._ddg_news(query, max_results)

        items: list[NewsItem] = []
        for article in data.get("news", []):
            title = article.get("title", "")
            if not title:
                continue
            items.append(
                NewsItem(
                    title=title,
                    url=article.get("link"),
                    snippet=article.get("snippet"),
                    source_name=article.get("source"),
                    published_date=article.get("date"),
                )
            )
        return items[:max_results]

    # ------------------------------------------------------------------
    # Backend: Brave Search
    # ------------------------------------------------------------------

    async def _brave_news(self, query: str, max_results: int) -> list[NewsItem]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_key,
        }
        params = {"q": query, "count": min(max_results, 20)}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/news/search",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Brave news search failed: %s", exc)
            return await self._ddg_news(query, max_results)

        items: list[NewsItem] = []
        for result in data.get("results", []):
            title = result.get("title", "")
            if not title:
                continue
            source_field = result.get("source")
            source_name = source_field.get("name") if isinstance(source_field, dict) else None
            items.append(
                NewsItem(
                    title=title,
                    url=result.get("url"),
                    snippet=result.get("description"),
                    source_name=source_name,
                    published_date=result.get("age"),
                )
            )
        return items[:max_results]

    # ------------------------------------------------------------------
    # Backend: DuckDuckGo HTML fallback
    # ------------------------------------------------------------------

    async def _ddg_news(self, query: str, max_results: int) -> list[NewsItem]:
        headers = {"User-Agent": _DEFAULT_USER_AGENT}
        params = {"q": query}
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/", params=params, headers=headers
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            logger.warning("DuckDuckGo news fallback failed: %s", exc)
            return []

        soup = BeautifulSoup(html, "lxml")
        items: list[NewsItem] = []
        for result in soup.select(".result__body")[:max_results]:
            title_tag = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            url = title_tag.get("href", "") or None
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
            items.append(NewsItem(title=title, url=url, snippet=snippet))
        return items
