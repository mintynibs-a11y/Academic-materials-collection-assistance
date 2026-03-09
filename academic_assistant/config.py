"""
Configuration management for the Academic Assistant.

Loads settings from environment variables or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the project root if it exists
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """Centralised configuration loaded from environment variables."""

    # ------------------------------------------------------------------ #
    # LLM provider settings                                               #
    # ------------------------------------------------------------------ #
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "anthropic" | "deepseek"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    # DeepSeek: https://platform.deepseek.com/
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # ------------------------------------------------------------------ #
    # Academic database API keys                                          #
    # ------------------------------------------------------------------ #
    # Web of Science (Clarivate): https://developer.clarivate.com/
    WOS_API_KEY: str = os.getenv("WOS_API_KEY", "")
    WOS_BASE_URL: str = "https://api.clarivate.com/apis/wos-starter/v1"

    # IEEE Xplore: https://developer.ieee.org/
    IEEE_API_KEY: str = os.getenv("IEEE_API_KEY", "")
    IEEE_BASE_URL: str = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    # ------------------------------------------------------------------ #
    # General web search                                                  #
    # ------------------------------------------------------------------ #
    # Serper (Google Search API): https://serper.dev/
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    # Brave Search API: https://api.search.brave.com/
    BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Search defaults                                                     #
    # ------------------------------------------------------------------ #
    DEFAULT_MAX_RESULTS: int = int(os.getenv("DEFAULT_MAX_RESULTS", "10"))
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")

    # ------------------------------------------------------------------ #
    # MCP server transport                                                #
    # ------------------------------------------------------------------ #
    MCP_TRANSPORT: str = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" | "sse"
    MCP_HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8765"))


config = Config()
