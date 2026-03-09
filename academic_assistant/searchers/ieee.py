"""
IEEE Xplore search adapter.

Uses the official IEEE Xplore REST API v1.
Documentation: https://developer.ieee.org/docs/read/IEEE_Xplore_API_Spec

Requires a valid IEEE_API_KEY environment variable.
"""

from __future__ import annotations

import logging

import httpx

from academic_assistant.models.paper import Paper, SearchResult, Source
from academic_assistant.searchers.base import BaseSearcher

logger = logging.getLogger(__name__)


class IEEESearcher(BaseSearcher):
    """Search adapter for IEEE Xplore."""

    source = Source.IEEE
    _BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        if not self.api_key:
            return self._error_result(
                query, "IEEE_API_KEY not set. Get a key at https://developer.ieee.org/"
            )

        params = {
            "apikey": self.api_key,
            "querytext": query,
            "max_records": min(max_results, 200),  # API maximum is 200
            "sort_field": "article_number",
            "sort_order": "desc",
            "output_type": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self._BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            return self._error_result(query, f"HTTP {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            return self._error_result(query, str(exc))

        papers = self._parse_response(data)
        total = data.get("total_records", len(papers))
        return SearchResult(query=query, source=self.source, papers=papers, total_found=total)

    def _parse_response(self, data: dict) -> list[Paper]:
        papers: list[Paper] = []
        for article in data.get("articles", []):
            authors = [a.get("full_name", "") for a in article.get("authors", {}).get("authors", [])]

            year_raw = article.get("publication_year")
            year: int | None = None
            if year_raw:
                try:
                    year = int(str(year_raw))
                except ValueError:
                    pass

            doi = article.get("doi")
            article_number = article.get("article_number", "")
            url = f"https://ieeexplore.ieee.org/document/{article_number}" if article_number else None

            keywords_raw = article.get("index_terms", {})
            keywords: list[str] = []
            for kw_group in keywords_raw.values():
                keywords.extend(kw_group.get("terms", []))

            papers.append(
                Paper(
                    title=article.get("title", ""),
                    authors=authors,
                    abstract=article.get("abstract"),
                    year=year,
                    journal=article.get("publication_title"),
                    doi=doi,
                    url=url,
                    citations=article.get("citing_paper_count"),
                    keywords=list(set(keywords)),
                    source=self.source,
                )
            )
        return papers
