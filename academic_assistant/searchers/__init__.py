"""Search adapters for various academic databases."""

from academic_assistant.searchers.base import BaseSearcher
from academic_assistant.searchers.web_of_science import WebOfScienceSearcher
from academic_assistant.searchers.google_scholar import GoogleScholarSearcher
from academic_assistant.searchers.cnki import CNKISearcher
from academic_assistant.searchers.ieee import IEEESearcher
from academic_assistant.searchers.web_search import WebSearcher
from academic_assistant.searchers.web_news import WebNewsSearcher

__all__ = [
    "BaseSearcher",
    "WebOfScienceSearcher",
    "GoogleScholarSearcher",
    "CNKISearcher",
    "IEEESearcher",
    "WebSearcher",
    "WebNewsSearcher",
]
