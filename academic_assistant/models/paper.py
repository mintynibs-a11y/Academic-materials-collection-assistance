"""
Core data models for papers, news items, and search results.

All models use Pydantic v2 for validation and serialisation.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Source(str, Enum):
    """Enumeration of supported academic search sources."""

    WEB_OF_SCIENCE = "web_of_science"
    GOOGLE_SCHOLAR = "google_scholar"
    CNKI = "cnki"
    IEEE = "ieee"
    ARXIV = "arxiv"
    WEB = "web"
    UNKNOWN = "unknown"


class Paper(BaseModel):
    """Represents a single academic paper or article."""

    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    year: Optional[int] = Field(None, description="Publication year", ge=1000)
    published_date: Optional[date] = Field(None, description="Full publication date")
    journal: Optional[str] = Field(None, description="Journal or conference name")
    journal_partition: Optional[str] = Field(
        None,
        description=(
            "Journal partition / indexing tier, e.g. 'SCI Q1', 'SCI Q2', "
            "'SCI Q3', 'SCI Q4', 'EI', 'ESCI', 'CSCD', '北大核心', '无分区'"
        ),
    )
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="Link to paper or landing page")
    pdf_url: Optional[str] = Field(None, description="Direct link to PDF if available")
    citations: Optional[int] = Field(None, description="Number of citations", ge=0)
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")
    source: Source = Field(Source.UNKNOWN, description="Database the record came from")
    relevance_score: Optional[float] = Field(
        None, description="LLM-assigned relevance score (0–1)", ge=0.0, le=1.0
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def formatted_authors(self) -> str:
        """Return a human-readable author string."""
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."

    def short_summary(self) -> str:
        """Return a concise one-line summary suitable for display lists."""
        year_str = f" ({self.year})" if self.year else ""
        citations_str = f", cited {self.citations}×" if self.citations is not None else ""
        return f"{self.title}{year_str} — {self.formatted_authors}{citations_str}"


class NewsItem(BaseModel):
    """A single research-news article scraped from the web."""

    title: str = Field(..., description="Article title")
    url: Optional[str] = Field(None, description="Link to the article")
    snippet: Optional[str] = Field(None, description="Short excerpt or description")
    source_name: Optional[str] = Field(None, description="Publishing site or outlet name")
    published_date: Optional[str] = Field(None, description="Publication date string as found on the page")


class SearchResult(BaseModel):
    """Container for search results from a single source."""

    query: str = Field(..., description="The original search query")
    source: Source = Field(..., description="Database that returned these results")
    papers: list[Paper] = Field(default_factory=list, description="Found papers")
    total_found: Optional[int] = Field(
        None, description="Total results reported by the source (may exceed returned count)"
    )
    error: Optional[str] = Field(None, description="Error message if the search failed")

    @property
    def success(self) -> bool:
        return self.error is None


class RankedPaper(BaseModel):
    """A paper that has been scored and ranked by the assistant."""

    paper: Paper
    rank: int = Field(..., description="1-based rank within the result set", ge=1)
    relevance_score: float = Field(
        ..., description="Overall relevance score assigned by the LLM (0–1)", ge=0.0, le=1.0
    )
    reason: str = Field(..., description="Short explanation of why this paper was ranked here")


class ResearchReport(BaseModel):
    """Final output produced by the assistant after a full research session."""

    query: str
    keywords: list[str] = Field(default_factory=list, description="Keywords used for the search")
    sources_searched: list[Source]
    total_papers_found: int
    ranked_papers: list[RankedPaper]
    summary: str = Field(..., description="LLM-generated narrative summary of the findings")
    key_themes: list[str] = Field(default_factory=list, description="Major themes identified")
    research_gaps: list[str] = Field(
        default_factory=list, description="Research gaps or future directions noted"
    )
    news_items: list[NewsItem] = Field(
        default_factory=list, description="Recent research-news articles related to the keywords"
    )
    news_summary: str = Field(
        default="", description="LLM-generated summary of the research-news articles"
    )
