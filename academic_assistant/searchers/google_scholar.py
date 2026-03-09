"""
Google Scholar search adapter.

Uses the `scholarly` library which scrapes Google Scholar.
No API key is required, but Google may temporarily block requests
from automated scrapers.  For production use, configure a proxy via
the SCHOLARLY_PROXY environment variable (Tor or a commercial proxy).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from academic_assistant.models.paper import Paper, SearchResult, Source
from academic_assistant.searchers.base import BaseSearcher

logger = logging.getLogger(__name__)


class GoogleScholarSearcher(BaseSearcher):
    """Search adapter for Google Scholar via the *scholarly* library."""

    source = Source.GOOGLE_SCHOLAR

    def __init__(self, api_key: str = "") -> None:
        super().__init__(api_key)
        self._scholarly_available: bool | None = None

    def _ensure_scholarly(self) -> bool:
        """Lazy-import scholarly and optionally configure a proxy."""
        if self._scholarly_available is not None:
            return self._scholarly_available
        try:
            import scholarly as _s  # noqa: F401

            proxy_url = os.getenv("SCHOLARLY_PROXY", "")
            if proxy_url:
                from scholarly import ProxyGenerator

                pg = ProxyGenerator()
                pg.SingleProxy(http=proxy_url, https=proxy_url)
                _s.scholarly.use_proxy(pg)
            self._scholarly_available = True
        except ImportError:
            logger.error("scholarly not installed. Run: pip install scholarly")
            self._scholarly_available = False
        return self._scholarly_available

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        if not self._ensure_scholarly():
            return self._error_result(query, "scholarly library not available")

        try:
            papers = await asyncio.to_thread(self._sync_search, query, max_results)
        except Exception as exc:
            return self._error_result(query, str(exc))

        return SearchResult(query=query, source=self.source, papers=papers)

    def _sync_search(self, query: str, max_results: int) -> list[Paper]:
        """Run the blocking scholarly search in a thread-pool executor."""
        from scholarly import scholarly

        search_iter = scholarly.search_pubs(query)
        papers: list[Paper] = []
        for i, result in enumerate(search_iter):
            if i >= max_results:
                break
            papers.append(self._parse_pub(result))
        return papers

    def _parse_pub(self, pub: dict[str, Any]) -> Paper:
        bib = pub.get("bib", {})
        authors_raw = bib.get("author", "")
        if isinstance(authors_raw, str):
            authors = [a.strip() for a in authors_raw.split(" and ")] if authors_raw else []
        else:
            authors = list(authors_raw)

        year_raw = bib.get("pub_year")
        year: int | None = None
        if year_raw:
            try:
                year = int(str(year_raw))
            except ValueError:
                pass

        return Paper(
            title=bib.get("title", ""),
            authors=authors,
            abstract=bib.get("abstract"),
            year=year,
            journal=bib.get("venue") or bib.get("journal"),
            url=pub.get("pub_url"),
            citations=pub.get("num_citations"),
            source=self.source,
        )
