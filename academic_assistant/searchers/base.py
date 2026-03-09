"""
Base class for all academic search adapters.

All searchers must implement the `search` coroutine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from academic_assistant.models.paper import SearchResult, Source

logger = logging.getLogger(__name__)


class BaseSearcher(ABC):
    """Abstract base class that all search adapters must extend."""

    source: Source  # Each subclass must declare its source enum value

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Execute a search and return a :class:`SearchResult`.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Maximum number of papers to return.

        Returns
        -------
        SearchResult
            Always returns a result object; sets ``error`` on failure
            rather than raising an exception so that callers can
            aggregate results from multiple sources gracefully.
        """

    def _error_result(self, query: str, error: str) -> SearchResult:
        """Build a failed :class:`SearchResult` with an error message."""
        logger.warning("[%s] search failed: %s", self.source, error)
        return SearchResult(query=query, source=self.source, error=error)
