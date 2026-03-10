"""
MCP Server – Academic Research Tools

Exposes the following tools to any MCP-compatible client (e.g. Claude Desktop):

  • search_web_of_science  – search Clarivate WoS
  • search_google_scholar  – search Google Scholar
  • search_cnki            – search 中国知网 CNKI
  • search_ieee            – search IEEE Xplore
  • search_web             – general web search
  • search_all_sources     – fan-out search across all sources simultaneously
  • rank_and_summarize     – rank collected papers and generate a summary via LLM

Run this server with:
    python -m academic_assistant.mcp_server.server
or via the entry-point defined in pyproject.toml/setup.cfg.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from academic_assistant.config import config
from academic_assistant.models.paper import ResearchReport, SearchResult, Source
from academic_assistant.searchers import (
    CNKISearcher,
    GoogleScholarSearcher,
    IEEESearcher,
    WebNewsSearcher,
    WebOfScienceSearcher,
    WebSearcher,
)

logger = logging.getLogger(__name__)

app = Server("academic-assistant")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_SEARCH_TOOLS: list[Tool] = [
    Tool(
        name="search_web_of_science",
        description=(
            "Search Web of Science (Clarivate) for academic papers. "
            "Requires WOS_API_KEY to be configured. "
            "Returns papers sorted by citation count."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10, max 50)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_google_scholar",
        description=(
            "Search Google Scholar for academic papers. "
            "No API key required. Google may rate-limit automated requests."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_cnki",
        description=(
            "Search 中国知网 (CNKI) for Chinese academic papers. "
            "No API key required. Set CNKI_SESSION_COOKIE for better access."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (Chinese or English)"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_ieee",
        description=(
            "Search IEEE Xplore for academic papers on engineering and computer science. "
            "Requires IEEE_API_KEY to be configured."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10, max 200)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_web",
        description=(
            "Perform a general web search for academic papers. "
            "Uses Serper, Brave Search, or DuckDuckGo as fallback. "
            "Set SERPER_API_KEY or BRAVE_API_KEY for best results."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_all_sources",
        description=(
            "Search ALL academic sources simultaneously (WoS, Google Scholar, CNKI, IEEE, Web). "
            "Aggregates and deduplicates results. Ideal for a comprehensive literature search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results_per_source": {
                    "type": "integer",
                    "description": "Max results per source (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rank_and_summarize",
        description=(
            "Given a list of papers (as returned by one of the search tools), "
            "rank them by relevance to the query and generate an LLM-powered summary. "
            "Returns a ResearchReport with ranked papers, key themes, and research gaps."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original research query for context",
                },
                "papers_json": {
                    "type": "string",
                    "description": (
                        "JSON array of paper objects as returned by a search tool. "
                        "Each object must have at least 'title'; 'abstract', 'authors', "
                        "'year', 'citations', 'journal_partition', and 'source' are used "
                        "for ranking when present."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many top papers to include in the ranked report (default 10)",
                    "default": 10,
                },
            },
            "required": ["query", "papers_json"],
        },
    ),
    Tool(
        name="search_news",
        description=(
            "Search the web for recent research-news articles related to a list of keywords. "
            "Returns news article titles, URLs, snippets, and an LLM-generated summary. "
            "Uses Serper, Brave Search, or DuckDuckGo as fallback."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search keywords",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of news items to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["keywords"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool listing handler
# ---------------------------------------------------------------------------


@app.list_tools()
async def list_tools() -> list[Tool]:
    return _SEARCH_TOOLS


# ---------------------------------------------------------------------------
# Tool call handler
# ---------------------------------------------------------------------------


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    query: str = arguments.get("query", "")
    max_results: int = int(arguments.get("max_results", config.DEFAULT_MAX_RESULTS))

    if name == "search_web_of_science":
        result = await WebOfScienceSearcher(config.WOS_API_KEY).search(query, max_results)
        return [TextContent(type="text", text=_format_search_result(result))]

    if name == "search_google_scholar":
        result = await GoogleScholarSearcher().search(query, max_results)
        return [TextContent(type="text", text=_format_search_result(result))]

    if name == "search_cnki":
        result = await CNKISearcher().search(query, max_results)
        return [TextContent(type="text", text=_format_search_result(result))]

    if name == "search_ieee":
        result = await IEEESearcher(config.IEEE_API_KEY).search(query, max_results)
        return [TextContent(type="text", text=_format_search_result(result))]

    if name == "search_web":
        result = await WebSearcher(config.SERPER_API_KEY).search(query, max_results)
        return [TextContent(type="text", text=_format_search_result(result))]

    if name == "search_all_sources":
        max_per = int(arguments.get("max_results_per_source", 5))
        results = await _search_all(query, max_per)
        combined_json = _combine_results_json(results)
        return [TextContent(type="text", text=combined_json)]

    if name == "rank_and_summarize":
        papers_json: str = arguments.get("papers_json", "[]")
        top_n: int = int(arguments.get("top_n", 10))
        report_text = await _rank_and_summarize(query, papers_json, top_n)
        return [TextContent(type="text", text=report_text)]

    if name == "search_news":
        keywords: list[str] = arguments.get("keywords", [])
        news_text = await _search_news(keywords, max_results)
        return [TextContent(type="text", text=news_text)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_search_result(result: SearchResult) -> str:
    """Convert a :class:`SearchResult` to a JSON string."""
    return result.model_dump_json(indent=2)


async def _search_all(query: str, max_per_source: int) -> list[SearchResult]:
    """Fan out search to all sources concurrently."""
    searchers = [
        WebOfScienceSearcher(config.WOS_API_KEY),
        GoogleScholarSearcher(),
        CNKISearcher(),
        IEEESearcher(config.IEEE_API_KEY),
        WebSearcher(config.SERPER_API_KEY),
    ]
    tasks = [s.search(query, max_per_source) for s in searchers]
    return list(await asyncio.gather(*tasks))


def _combine_results_json(results: list[SearchResult]) -> str:
    """Merge multiple SearchResults into a single JSON representation."""
    all_papers = []
    errors: list[str] = []
    for r in results:
        if r.error:
            errors.append(f"{r.source}: {r.error}")
        for paper in r.papers:
            all_papers.append(paper.model_dump())

    return json.dumps(
        {
            "total_papers": len(all_papers),
            "errors": errors,
            "papers": all_papers,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _rank_and_summarize(query: str, papers_json: str, top_n: int) -> str:
    """Use the configured LLM to rank and summarise papers."""
    # Import the processor lazily to avoid hard dependency on LLM keys at server start-up
    from academic_assistant.processors.ranker import PaperRanker

    try:
        papers_data = json.loads(papers_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid papers_json: {exc}"})

    ranker = PaperRanker()
    report = await ranker.rank_and_summarize(query, papers_data, top_n)
    return report.model_dump_json(indent=2, ensure_ascii=False)


async def _search_news(keywords: list[str], max_results: int) -> str:
    """Scrape research-news articles and return as JSON."""
    from academic_assistant.processors.ranker import PaperRanker

    news_searcher = WebNewsSearcher()
    items = await news_searcher.search_news(keywords, max_results)
    ranker = PaperRanker()
    summary = await ranker.summarize_news(keywords, items)
    return json.dumps(
        {
            "keywords": keywords,
            "total_news": len(items),
            "news_summary": summary,
            "news_items": [item.model_dump() for item in items],
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
