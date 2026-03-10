"""
Academic Assistant - high-level client

This module provides the :class:`AcademicAssistant` class that orchestrates
a full research session:

1. Accepts a list of specific keywords (or a natural-language query).
2. Fans out searches to all configured academic sources concurrently.
3. Annotates each paper with its journal partition tier.
4. Aggregates and deduplicates the results.
5. Calls the LLM to rank the papers and generate a research report.
6. Scrapes recent research-news articles for the keywords and adds an
   LLM-generated news summary to the report.
7. Returns and optionally prints a structured :class:`ResearchReport`.

Usage (CLI):

    python main.py --keywords "deep learning" "protein folding"

Usage (programmatic):

    import asyncio
    from academic_assistant.assistant import AcademicAssistant

    async def main():
        assistant = AcademicAssistant()
        report = await assistant.research(
            keywords=["deep learning", "protein folding"],
            top_n=15,
        )
        AcademicAssistant.print_report(report)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

from academic_assistant.config import config
from academic_assistant.models.paper import NewsItem, Paper, ResearchReport, SearchResult, Source
from academic_assistant.processors.ranker import PaperRanker
from academic_assistant.searchers import (
    CNKISearcher,
    GoogleScholarSearcher,
    IEEESearcher,
    WebNewsSearcher,
    WebOfScienceSearcher,
    WebSearcher,
)
from academic_assistant.utils.journal_partition import lookup_partition

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

    async def research(
        self,
        query: str = "",
        keywords: Optional[list[str]] = None,
        top_n: int = 10,
        include_news: bool = True,
    ) -> ResearchReport:
        """
        Run a complete research session.

        Parameters
        ----------
        query:
            Natural-language research topic or question (used when *keywords*
            is not supplied).
        keywords:
            Specific search keywords.  When provided these are used instead of
            *query*; the academic search uses ``keyword1 AND keyword2 ...``
            syntax and the display query is set to the joined keywords.
        top_n:
            Number of top papers to include in the ranked report.
        include_news:
            Whether to also scrape research-news articles and add an LLM
            summary of them to the report.

        Returns
        -------
        ResearchReport
        """
        # Build the search query
        if keywords:
            search_query = " AND ".join(keywords)
            display_query = "; ".join(keywords)
        else:
            search_query = query
            display_query = query
            keywords = []

        logger.info("Starting research session for: %s", display_query)

        # 1. Search all academic sources concurrently
        search_results = await self._search_all(search_query)

        # 2. Aggregate, deduplicate, and annotate with journal partitions
        papers = self._aggregate(search_results)
        logger.info(
            "Aggregated %d unique papers from %d sources",
            len(papers),
            len(search_results),
        )

        # 3. Rank and summarise via LLM
        papers_data = [p.model_dump() for p in papers]
        report = await self._ranker.rank_and_summarize(display_query, papers_data, top_n)
        # Attach the keywords list to the report
        report = report.model_copy(update={"keywords": keywords})

        # 4. Optionally fetch and summarise research news
        if include_news and keywords:
            news_items = await self._fetch_news(keywords)
            news_summary = await self._ranker.summarize_news(keywords, news_items)
            report = report.model_copy(
                update={"news_items": news_items, "news_summary": news_summary}
            )

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
        Merge papers from multiple sources, remove near-duplicates, and
        annotate each paper with its journal partition.

        Deduplication is based on normalised title comparison.
        """
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for result in results:
            if result.error:
                logger.warning(
                    "Source %s returned error: %s", result.source, result.error
                )
                continue
            for paper in result.papers:
                norm = paper.title.lower().strip()
                if norm and norm not in seen_titles:
                    seen_titles.add(norm)
                    # Annotate journal partition if not already set
                    if paper.journal_partition is None and paper.journal:
                        partition = lookup_partition(paper.journal)
                        if partition is not None:
                            paper = paper.model_copy(
                                update={"journal_partition": partition}
                            )
                    unique.append(paper)

        return unique

    async def _fetch_news(self, keywords: list[str]) -> list[NewsItem]:
        """Scrape research-news articles for the given keywords."""
        try:
            news_searcher = WebNewsSearcher()
            return await news_searcher.search_news(
                keywords, max_results=config.DEFAULT_MAX_RESULTS
            )
        except Exception as exc:
            logger.warning("News search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: ResearchReport) -> None:
        """Pretty-print a :class:`ResearchReport` to stdout."""
        separator = "=" * 70
        print(separator)
        print(f"RESEARCH REPORT: {report.query}")
        if report.keywords:
            print(f"Keywords: {', '.join(report.keywords)}")
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
                partition_str = (
                    f"  [{p.journal_partition}]" if p.journal_partition else ""
                )
                print(f"       Journal : {p.journal}{partition_str}")
            if p.abstract:
                abstract_preview = p.abstract[:200].replace("\n", " ")
                suffix = "..." if len(p.abstract) > 200 else ""
                print(f"       Abstract: {abstract_preview}{suffix}")
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
                print(f"  * {theme}")

        if report.research_gaps:
            print("\n--- RESEARCH GAPS / FUTURE DIRECTIONS ---")
            for gap in report.research_gaps:
                print(f"  * {gap}")

        print("\n--- SUMMARY ---")
        print(report.summary)

        if report.news_items:
            print(f"\n--- RESEARCH NEWS ({len(report.news_items)} articles) ---")
            for item in report.news_items:
                print(f"\n  * {item.title}")
                if item.source_name:
                    print(f"    Source : {item.source_name}")
                if item.published_date:
                    print(f"    Date   : {item.published_date}")
                if item.url:
                    print(f"    URL    : {item.url}")
                if item.snippet:
                    print(f"    Snippet: {item.snippet[:150]}...")

        if report.news_summary:
            print("\n--- NEWS SUMMARY ---")
            print(report.news_summary)

        print(separator)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _cli_main(
    query: str,
    keywords: list[str],
    top_n: int,
    max_per_source: int,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    assistant = AcademicAssistant(max_results_per_source=max_per_source)
    if keywords:
        report = await assistant.research(keywords=keywords, top_n=top_n)
    else:
        report = await assistant.research(query=query, top_n=top_n)
    AcademicAssistant.print_report(report)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Academic Research Assistant - search, rank, and summarise papers"
    )
    # Accept either a positional free-text query or --keywords
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Research topic or question (use --keywords for keyword-based search)",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        metavar="KW",
        default=[],
        help=(
            "One or more specific keywords for the search "
            "(e.g. --keywords 'machine learning' 'medical imaging')"
        ),
    )
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

    if not args.query and not args.keywords:
        parser.error("Provide either a positional query or --keywords")

    asyncio.run(
        _cli_main(args.query, args.keywords, args.top_n, args.max_per_source)
    )


if __name__ == "__main__":
    main()
