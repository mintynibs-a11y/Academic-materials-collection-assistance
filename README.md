# Academic Materials Collection Assistance

> 帮我从 Web of Science、谷歌学术、中国知网、IEEE 美国电气电子工程师学会与互联网上搜集最新的学术文章资料，并进行挑选、排序、整理、总结

[中文文档](README_zh.md)

An AI-powered academic research assistant that searches multiple academic databases simultaneously, ranks results by relevance, and generates a comprehensive research summary using large language model APIs and the **Model Context Protocol (MCP)**.

---

## Features

| Capability | Details |
|---|---|
| **Multi-source search** | Web of Science · Google Scholar · 中国知网 (CNKI) · IEEE Xplore · General web |
| **Concurrent search** | All sources queried in parallel for fast results |
| **Deduplication** | Near-duplicate papers removed automatically |
| **Journal partition** | Each paper is annotated with its tier (SCI Q1–Q4 / EI / ESCI / CSCD / 北大核心) |
| **LLM ranking** | Papers ranked 0–1 by relevance using OpenAI, Anthropic, or DeepSeek; journal tier and citation count are factored in |
| **Research summary** | Narrative summary, key themes, and research gaps |
| **Research news** | Scrapes recent research-news articles and generates an LLM summary |
| **MCP server** | Expose all tools to Claude Desktop or any MCP client |
| **CLI** | Run a full research session from the command line (`--keywords` or free-text query) |

---

## Architecture

```
academic_assistant/
├── config.py                  # Environment-based configuration
├── assistant.py               # High-level orchestrator + CLI entry point
├── models/
│   └── paper.py               # Pydantic models: Paper, NewsItem, SearchResult, ResearchReport
├── searchers/
│   ├── base.py                # Abstract BaseSearcher
│   ├── web_of_science.py      # Clarivate WoS REST API
│   ├── google_scholar.py      # scholarly library (scraping)
│   ├── cnki.py                # CNKI web scraping
│   ├── ieee.py                # IEEE Xplore REST API
│   ├── web_search.py          # Serper / Brave / DuckDuckGo fallback
│   └── web_news.py            # Research-news scraping (appends to web search)
├── processors/
│   └── ranker.py              # LLM-powered ranking, summarisation & news summary
├── utils/
│   └── journal_partition.py   # Static journal → partition-tier lookup table
└── mcp_server/
    └── server.py              # MCP server exposing 8 tools (incl. search_news)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/mintynibs-a11y/Academic-materials-collection-assistance.git
cd Academic-materials-collection-assistance
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in the keys you have
```

| Variable | Source | Required? |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Yes (or use Anthropic / DeepSeek) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Alt. to OpenAI |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) | Alt. to OpenAI |
| `WOS_API_KEY` | [developer.clarivate.com](https://developer.clarivate.com) | Optional |
| `IEEE_API_KEY` | [developer.ieee.org](https://developer.ieee.org) | Optional |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) | Optional (improves web search) |
| `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) | Optional |
| `CNKI_SESSION_COOKIE` | Browser DevTools after login to cnki.net | Optional |

### 3. Run the CLI assistant

```bash
# Basic research query (free-text)
python main.py "large language models in clinical medicine"

# Keyword-based search (AND-joined; more precise)
python main.py --keywords "large language models" "clinical medicine"

# Customise result count
python main.py "transformer architecture" --top-n 20 --max-per-source 15
python main.py --keywords "transformer" "architecture" --top-n 20 --max-per-source 15
```

### 4. Use as MCP server (Claude Desktop integration)

Start the MCP server over stdio (for Claude Desktop):

```bash
python main.py --mcp-server
```

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "academic-assistant": {
      "command": "python",
      "args": ["/path/to/Academic-materials-collection-assistance/main.py", "--mcp-server"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "WOS_API_KEY": "...",
        "IEEE_API_KEY": "..."
      }
    }
  }
}
```

### 5. Programmatic usage

```python
import asyncio
from academic_assistant.assistant import AcademicAssistant

async def main():
    assistant = AcademicAssistant(max_results_per_source=10)
    # Keyword-based search (recommended for precision)
    report = await assistant.research(
        keywords=["deep learning", "protein structure prediction"],
        top_n=15,
    )
    AcademicAssistant.print_report(report)

asyncio.run(main())
```

---

## MCP Tools

When running as an MCP server the following tools are available:

| Tool | Description |
|---|---|
| `search_web_of_science` | Search Clarivate WoS (requires `WOS_API_KEY`) |
| `search_google_scholar` | Search Google Scholar (no key needed) |
| `search_cnki` | Search 中国知网 CNKI |
| `search_ieee` | Search IEEE Xplore (requires `IEEE_API_KEY`) |
| `search_web` | General web search (Serper / Brave / DuckDuckGo) |
| `search_all_sources` | Fan-out to all sources simultaneously |
| `rank_and_summarize` | LLM ranking + research report generation |
| `search_news` | Scrape research-news articles and return an LLM summary |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
