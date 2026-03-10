"""
Web of Science search adapter.

Uses the Clarivate Web of Science Starter API v1.
Documentation: https://developer.clarivate.com/apis/wos-starter

Requires a valid WOS_API_KEY environment variable.
"""

from __future__ import annotations

import logging

import httpx

from academic_assistant.models.paper import Paper, SearchResult, Source
from academic_assistant.searchers.base import BaseSearcher

logger = logging.getLogger(__name__)


class WebOfScienceSearcher(BaseSearcher):
    """Search adapter for Web of Science (Clarivate)."""

    source = Source.WEB_OF_SCIENCE
    _BASE_URL = "https://api.clarivate.com/apis/wos-starter/v1"

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        if not self.api_key:
            return self._error_result(
                query, "WOS_API_KEY not set. Get a key at https://developer.clarivate.com/"
            )

        params = {
            "q": query,
            "limit": min(max_results, 50),  # API maximum per request is 50
            "page": 1,
            "db": "WOS",
            "sortField": "TC",  # sort by times-cited descending
        }
        headers = {"X-ApiKey": self.api_key, "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self._BASE_URL}/documents", params=params, headers=headers
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            return self._error_result(query, f"HTTP {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            return self._error_result(query, str(exc))

        papers = self._parse_response(data)
        total = data.get("metadata", {}).get("total", len(papers))
        return SearchResult(
            query=query,
            source=self.source,
            papers=papers,
            total_found=total,
        )

    def _parse_response(self, data: dict) -> list[Paper]:
        papers: list[Paper] = []
        for hit in data.get("hits", []):
            names = hit.get("names", {})
            authors = [
                a.get("displayName", "")
                for a in names.get("authors", [])
            ]
            source_info = hit.get("source", {})
            identifiers = {i["type"]: i["value"] for i in hit.get("identifiers", [])}

            # Publication date / year
            pub_date = source_info.get("publishDate", "")
            year: int | None = None
            if pub_date and len(pub_date) >= 4:
                try:
                    year = int(pub_date[:4])
                except ValueError:
                    pass

            papers.append(
                Paper(
                    title=hit.get("title", ""),
                    authors=authors,
                    abstract=hit.get("abstract", None),
                    year=year,
                    journal=source_info.get("sourceTitle"),
                    doi=identifiers.get("doi"),
                    url=f"https://www.webofscience.com/wos/woscc/full-record/{hit.get('uid', '')}",
                    citations=hit.get("citations", [{}])[0].get("count") if hit.get("citations") else None,
                    keywords=[kw.get("value", "") for kw in hit.get("keywords", {}).get("authorKeywords", [])],
                    source=self.source,
                )
            )
        return papers
