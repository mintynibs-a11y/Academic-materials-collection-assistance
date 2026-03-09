"""
Academic Assistant – high-level client

This module provides the :class:`AcademicAssistant` class that orchestrates
a full research session:

1. Accepts a natural-language research query.
2. Fans out searches to all configured academic sources concurrently.
3. Aggregates and deduplicates the results.
4. Calls the LLM to rank the papers and generate a research report.
5. Returns and optionally prints a structured :class:`ResearchReport`.

Usage (CLI / interactive):

    python -m academic_assistant.assistant "large language models in medicine"

Usage (programmatic):

    import asyncio
    from academic_assistant.assistant import AcademicAssistant

    async def main():
        assistant = AcademicAssistant()
        report = await assistant.research("deep learning for protein folding", top_n=15)
        print(report.summary)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

from academic_assistant.config import config
from academic_assistant.models.paper import Paper, ResearchReport, SearchResult, Source
from academic_assistant.processors.ranker import PaperRanker
from academic_assistant.searchers import (
    CNKISearcher,
    GoogleScholarSearcher,
    IEEESearcher,
    WebOfScienceSearcher,
    WebSearcher,
)

logger = logging.getLogger(__name__)


class AcademicAssistant:
    """
    Orchestrates a full research session across multiple academic databases.

    Parameters
    ----------
    max_results_per_source:
        How many papers to request from each database.
    sources:
        Which databases to search. Defaults to all five.
    """

    def __init__(
        self,
        max_results_per_source: int = 10,
        sources: Optional[list[Source]] = None,
    ) -> None:
        self.max_results_per_source = max_results_per_source
        self.enabled_sources = sources or list(Source)
        self._ranker = PaperRanker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research(self, query: str, top_n: int = 10) -> ResearchReport:
        """
        Run a complete research session for *query*.

        Parameters
        ----------
        query:
            Natural language research topic or question.
        top_n:
            Number of top papers to include in the ranked report.

        Returns
        -------
        ResearchReport
        """
        logger.info("Starting research session for: %s", query)

        # 1. Search all sources concurrently
        search_results = await self._search_all(query)

        # 2. Aggregate & deduplicate
        papers = self._aggregate(search_results)
        logger.info("Aggregated %d unique papers from %d sources", len(papers), len(search_results))

        # 3. Rank and summarise via LLM
        papers_data = [p.model_dump() for p in papers]
        report = await self._ranker.rank_and_summarize(query, papers_data, top_n)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_all(self, query: str) -> list[SearchResult]:
        """Fan-out search to all enabled sources."""
        searchers = []
        if Source.WEB_OF_SCIENCE in self.enabled_sources:
            searchers.append(WebOfScienceSearcher(config.WOS_API_KEY))
        if Source.GOOGLE_SCHOLAR in self.enabled_sources:
            searchers.append(GoogleScholarSearcher())
        if Source.CNKI in self.enabled_sources:
            searchers.append(CNKISearcher())
        if Source.IEEE in self.enabled_sources:
            searchers.append(IEEESearcher(config.IEEE_API_KEY))
        if Source.WEB in self.enabled_sources:
            searchers.append(WebSearcher(config.SERPER_API_KEY))

        tasks = [s.search(query, self.max_results_per_source) for s in searchers]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    def _aggregate(results: list[SearchResult]) -> list[Paper]:
        """
        Merge papers from multiple sources and remove near-duplicates.

        Deduplication is based on normalised title comparison.
        """
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for result in results:
            if result.error:
                logger.warning("Source %s returned error: %s", result.source, result.error)
                continue
            for paper in result.papers:
                norm = paper.title.lower().strip()
                if norm and norm not in seen_titles:
                    seen_titles.add(norm)
                    unique.append(paper)

        return unique

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: ResearchReport) -> None:
        """Pretty-print a :class:`ResearchReport` to stdout."""
        separator = "=" * 70
        print(separator)
        print(f"RESEARCH REPORT: {report.query}")
        print(separator)
        print(f"\nSources searched: {', '.join(str(s) for s in report.sources_searched)}")
        print(f"Total papers found: {report.total_papers_found}")
        print(f"Papers ranked: {len(report.ranked_papers)}")

        print("\n--- RANKED PAPERS ---")
        for rp in report.ranked_papers:
            p = rp.paper
            print(f"\n  #{rp.rank}  [{rp.relevance_score:.2f}]  {p.title}")
            print(f"       Authors : {p.formatted_authors}")
            if p.year:
                print(f"       Year    : {p.year}")
            if p.journal:
                print(f"       Journal : {p.journal}")
            if p.citations is not None:
                print(f"       Citations: {p.citations}")
            if p.doi:
                print(f"       DOI     : {p.doi}")
            if p.url:
                print(f"       URL     : {p.url}")
            print(f"       Reason  : {rp.reason}")

        if report.key_themes:
            print("\n--- KEY THEMES ---")
            for theme in report.key_themes:
                print(f"  • {theme}")

        if report.research_gaps:
            print("\n--- RESEARCH GAPS / FUTURE DIRECTIONS ---")
            for gap in report.research_gaps:
                print(f"  • {gap}")

        print("\n--- SUMMARY ---")
        print(report.summary)
        print(separator)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _cli_main(query: str, top_n: int, max_per_source: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    assistant = AcademicAssistant(max_results_per_source=max_per_source)
    report = await assistant.research(query, top_n=top_n)
    AcademicAssistant.print_report(report)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Academic Research Assistant – search, rank, and summarise papers"
    )
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top papers to include in the report (default: 10)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=config.DEFAULT_MAX_RESULTS,
        help=f"Max results to fetch per source (default: {config.DEFAULT_MAX_RESULTS})",
    )
    args = parser.parse_args()
    asyncio.run(_cli_main(args.query, args.top_n, args.max_per_source))


if __name__ == "__main__":
    main()
