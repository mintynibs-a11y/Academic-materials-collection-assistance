"""
CNKI (中国知网) search adapter.

CNKI does not provide a public open API, so this adapter scrapes the
public search page at https://kns.cnki.net/kns8s/defaultresult/index

Usage notes
-----------
* For best results, set a CNKI_SESSION_COOKIE env var with a valid
  CNKI session cookie obtained after logging in to cnki.net.
* An optional CNKI_PROXY env var may be set to route requests through
  a proxy (format: "http://user:pass@host:port").
* The adapter uses BeautifulSoup to parse the HTML response.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
from urllib.parse import quote

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
_CNKI_SEARCH_URL = "https://kns.cnki.net/kns8s/defaultresult/index"


class CNKISearcher(BaseSearcher):
    """Search adapter for 中国知网 (CNKI)."""

    source = Source.CNKI

    def __init__(self, api_key: str = "") -> None:
        super().__init__(api_key)
        self._session_cookie = os.getenv("CNKI_SESSION_COOKIE", "")
        self._proxy = os.getenv("CNKI_PROXY", None)

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        params = {
            "kw": query,
            "korder": "ST",
            "SortType": "pubdate",
        }
        headers = {
            "User-Agent": _DEFAULT_USER_AGENT,
            "Referer": "https://kns.cnki.net/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        if self._session_cookie:
            headers["Cookie"] = self._session_cookie

        proxies = {"http://": self._proxy, "https://": self._proxy} if self._proxy else None

        try:
            async with httpx.AsyncClient(
                timeout=30, proxies=proxies, follow_redirects=True
            ) as client:
                response = await client.get(_CNKI_SEARCH_URL, params=params, headers=headers)
                response.raise_for_status()
        except Exception as exc:
            return self._error_result(query, f"CNKI request failed: {exc}")

        try:
            papers = self._parse_html(response.text, max_results)
        except Exception as exc:
            return self._error_result(query, f"CNKI parsing failed: {exc}")

        return SearchResult(query=query, source=self.source, papers=papers)

    # ------------------------------------------------------------------
    # HTML parsing helpers
    # ------------------------------------------------------------------

    def _parse_html(self, html: str, max_results: int) -> list[Paper]:
        soup = BeautifulSoup(html, "lxml")
        papers: list[Paper] = []

        # CNKI search result items are in <div class="result-table-list"> → <tbody> → <tr>
        table = soup.find("div", class_="result-table-list")
        if not table:
            # Fallback: look for individual result cards
            rows = soup.select("tr.odd, tr.even")
        else:
            rows = table.select("tr")

        for row in rows[:max_results]:
            paper = self._parse_row(row)
            if paper:
                papers.append(paper)

        return papers

    def _parse_row(self, row: Any) -> Paper | None:
        title_tag = row.select_one("td.name a, .fz14")
        if not title_tag:
            return None
        title = title_tag.get_text(strip=True)
        if not title:
            return None

        href = title_tag.get("href", "")
        url = (_CNKI_DETAIL_BASE + href) if href.startswith("/") else href or None

        # Authors (typically comma/semicolon separated in a span)
        author_tag = row.select_one("td.author, .author")
        author_text = author_tag.get_text(strip=True) if author_tag else ""
        authors = [a.strip() for a in re.split(r"[,;，；]", author_text) if a.strip()]

        # Source / journal
        source_tag = row.select_one("td.source a, .source a")
        journal = source_tag.get_text(strip=True) if source_tag else None

        # Year
        date_tag = row.select_one("td.date, .date")
        date_text = date_tag.get_text(strip=True) if date_tag else ""
        year_match = re.search(r"(\d{4})", date_text)
        year = int(year_match.group(1)) if year_match else None

        # Citations
        cite_tag = row.select_one("td.quote, .quote")
        citations: int | None = None
        if cite_tag:
            cite_match = re.search(r"(\d+)", cite_tag.get_text())
            if cite_match:
                citations = int(cite_match.group(1))

        return Paper(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            url=url,
            citations=citations,
            source=self.source,
        )
