"""
LLM-powered paper ranking and summarisation.

Supports OpenAI, Anthropic, and DeepSeek as LLM backends.
The backend is selected based on the LLM_PROVIDER config value
("openai", "anthropic", or "deepseek").
"""

from __future__ import annotations

import json
import logging
from typing import Any

from academic_assistant.config import config
from academic_assistant.models.paper import NewsItem, Paper, RankedPaper, ResearchReport, Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates – paper ranking
# ---------------------------------------------------------------------------

_RANKING_SYSTEM_PROMPT = """\
You are an expert academic librarian and research analyst.
Your task is to evaluate a list of academic papers and rank them by their \
relevance to the user's research query.

For each paper, assign:
- relevance_score: a float between 0.0 (not relevant) and 1.0 (highly relevant)
- reason: a 1-2 sentence explanation of why you assigned that score

When computing the relevance_score, weight the following factors:
1. Topical relevance to the research query (primary factor).
2. Journal quality: SCI Q1 > SCI Q2 > SCI Q3 > SCI Q4 > EI/ESCI > core/CSCD > unknown.
3. Citation count: higher citations indicate greater influence.
4. Recency: more recent papers are generally preferred when relevance is equal.

Then, after ranking all papers:
- Write a comprehensive narrative summary (3-5 paragraphs) of the overall state of \
research in this area.
- Identify 3-5 key themes across the papers.
- Identify 2-3 research gaps or future directions suggested by the literature.

Respond ONLY with valid JSON in exactly this format:
{
  "ranked_papers": [
    {
      "index": <0-based index into the input papers array>,
      "relevance_score": <float 0.0-1.0>,
      "reason": "<1-2 sentence explanation>"
    },
    ...
  ],
  "summary": "<narrative summary>",
  "key_themes": ["<theme 1>", "<theme 2>", ...],
  "research_gaps": ["<gap 1>", "<gap 2>", ...]
}
"""

_RANKING_USER_TEMPLATE = """\
Research query: {query}

Papers to evaluate (JSON array):
{papers_json}

Rank up to {top_n} of the most relevant papers.
"""

# ---------------------------------------------------------------------------
# Prompt templates – news summarisation
# ---------------------------------------------------------------------------

_NEWS_SUMMARY_SYSTEM_PROMPT = """\
You are a science journalist assistant. You will be given a list of recent \
research-news articles related to a set of keywords.

Your tasks:
1. Write a concise summary (2-3 paragraphs) of the current research landscape \
   and hot topics as reflected by the news articles.
2. List the 3-5 most important or interesting articles with a one-sentence \
   description of each.

Respond ONLY with valid JSON in exactly this format:
{
  "summary": "<narrative summary>",
  "highlights": [
    {"title": "<article title>", "url": "<article url or null>", "description": "<one sentence>"},
    ...
  ]
}
"""

_NEWS_SUMMARY_USER_TEMPLATE = """\
Keywords: {keywords}

Recent research-news articles (JSON array):
{news_json}
"""


class PaperRanker:
    """Ranks and summarises papers using a configured LLM backend."""

    def __init__(self) -> None:
        self._provider = config.LLM_PROVIDER.lower()

    async def rank_and_summarize(
        self, query: str, papers_data: list[dict[str, Any]], top_n: int = 10
    ) -> ResearchReport:
        """
        Rank *papers_data* by relevance to *query* and generate a research summary.

        Parameters
        ----------
        query:
            The original research query used for context.
        papers_data:
            List of paper dicts (as returned by the search tools / ``model_dump()``).
        top_n:
            Maximum number of ranked papers to return.

        Returns
        -------
        ResearchReport
        """
        papers = [self._dict_to_paper(p) for p in papers_data]

        if not papers:
            return ResearchReport(
                query=query,
                sources_searched=[],
                total_papers_found=0,
                ranked_papers=[],
                summary="No papers were provided for ranking.",
            )

        # Build a compact representation of each paper for the LLM
        compact = [
            {
                "index": i,
                "title": p.title,
                "authors": p.formatted_authors,
                "year": p.year,
                "journal": p.journal,
                "journal_partition": p.journal_partition,
                "abstract": (p.abstract or "")[:500],  # truncate long abstracts
                "citations": p.citations,
                "keywords": p.keywords[:10],
                "source": p.source,
            }
            for i, p in enumerate(papers)
        ]

        llm_json = await self._call_llm(query, compact, top_n)

        ranked_papers: list[RankedPaper] = []
        for rank_pos, item in enumerate(llm_json.get("ranked_papers", [])[:top_n], start=1):
            idx = item.get("index", 0)
            if idx < len(papers):
                ranked_papers.append(
                    RankedPaper(
                        paper=papers[idx],
                        rank=rank_pos,
                        relevance_score=float(item.get("relevance_score", 0.5)),
                        reason=item.get("reason", ""),
                    )
                )

        sources_searched = list({p.source for p in papers})

        return ResearchReport(
            query=query,
            sources_searched=sources_searched,
            total_papers_found=len(papers),
            ranked_papers=ranked_papers,
            summary=llm_json.get("summary", ""),
            key_themes=llm_json.get("key_themes", []),
            research_gaps=llm_json.get("research_gaps", []),
        )

    async def summarize_news(
        self, keywords: list[str], news_items: list[NewsItem]
    ) -> str:
        """
        Generate a narrative summary of *news_items* using the configured LLM.

        Parameters
        ----------
        keywords:
            The keywords used for the research session.
        news_items:
            List of news articles to summarise.

        Returns
        -------
        str
            Narrative summary, or an empty string if no LLM key is configured
            or *news_items* is empty.
        """
        if not news_items:
            return ""

        compact = [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "source_name": item.source_name,
                "published_date": item.published_date,
            }
            for item in news_items
        ]
        user_message = _NEWS_SUMMARY_USER_TEMPLATE.format(
            keywords=", ".join(keywords),
            news_json=json.dumps(compact, ensure_ascii=False, indent=2),
        )

        if self._provider == "anthropic":
            result = await self._call_anthropic_raw(_NEWS_SUMMARY_SYSTEM_PROMPT, user_message)
        elif self._provider == "deepseek":
            result = await self._call_deepseek_raw(_NEWS_SUMMARY_SYSTEM_PROMPT, user_message)
        else:
            result = await self._call_openai_raw(_NEWS_SUMMARY_SYSTEM_PROMPT, user_message)

        return result.get("summary", "")

    # ------------------------------------------------------------------
    # LLM call helpers (paper ranking)
    # ------------------------------------------------------------------

    async def _call_llm(
        self, query: str, compact_papers: list[dict], top_n: int
    ) -> dict[str, Any]:
        """Call the configured LLM and return the parsed JSON response."""
        user_message = _RANKING_USER_TEMPLATE.format(
            query=query,
            papers_json=json.dumps(compact_papers, ensure_ascii=False, indent=2),
            top_n=top_n,
        )

        if self._provider == "anthropic":
            return await self._call_anthropic(user_message)
        if self._provider == "deepseek":
            return await self._call_deepseek(user_message)
        return await self._call_openai(user_message)

    async def _call_openai(self, user_message: str) -> dict[str, Any]:
        return await self._call_openai_raw(_RANKING_SYSTEM_PROMPT, user_message)

    async def _call_anthropic(self, user_message: str) -> dict[str, Any]:
        return await self._call_anthropic_raw(_RANKING_SYSTEM_PROMPT, user_message)

    async def _call_deepseek(self, user_message: str) -> dict[str, Any]:
        return await self._call_deepseek_raw(_RANKING_SYSTEM_PROMPT, user_message)

    # ------------------------------------------------------------------
    # Raw LLM call helpers (reusable with any system prompt)
    # ------------------------------------------------------------------

    async def _call_openai_raw(
        self, system_prompt: str, user_message: str
    ) -> dict[str, Any]:
        if not config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set; returning empty ranking.")
            return {}
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            text = response.choices[0].message.content or "{}"
            return json.loads(text)
        except Exception as exc:
            logger.error("OpenAI call failed: %s", exc)
            return {}

    async def _call_anthropic_raw(
        self, system_prompt: str, user_message: str
    ) -> dict[str, Any]:
        if not config.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set; returning empty ranking.")
            return {}
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
            response = await client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.2,
            )
            text = response.content[0].text if response.content else "{}"
            # Strip markdown code fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as exc:
            logger.error("Anthropic call failed: %s", exc)
            return {}

    async def _call_deepseek_raw(
        self, system_prompt: str, user_message: str
    ) -> dict[str, Any]:
        """Call the DeepSeek API using its OpenAI-compatible interface."""
        if not config.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not set; returning empty ranking.")
            return {}
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
            )
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            text = response.choices[0].message.content or "{}"
            return json.loads(text)
        except Exception as exc:
            logger.error("DeepSeek call failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_paper(data: dict[str, Any]) -> Paper:
        """Convert a raw dict to a :class:`Paper`, tolerating missing fields."""
        try:
            return Paper.model_validate(data)
        except Exception:
            return Paper(title=data.get("title", "Unknown"), source=Source.UNKNOWN)
