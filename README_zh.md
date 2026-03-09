# 学术文献收集助手（中文说明文档）

> 从 Web of Science、谷歌学术、中国知网、IEEE Xplore 与互联网上自动搜集最新学术文章，并借助大语言模型（LLM）进行挑选、排序、整理与总结。

[English README](README.md)

---

## 目录

1. [项目简介](#1-项目简介)
2. [功能特性](#2-功能特性)
3. [系统架构](#3-系统架构)
4. [环境要求](#4-环境要求)
5. [安装步骤](#5-安装步骤)
6. [配置说明](#6-配置说明)
7. [快速上手](#7-快速上手)
   - [命令行（CLI）](#71-命令行cli)
   - [MCP 服务器与 Claude Desktop 集成](#72-mcp-服务器与-claude-desktop-集成)
   - [Python 编程接口](#73-python-编程接口)
8. [MCP 工具参考](#8-mcp-工具参考)
9. [数据模型说明](#9-数据模型说明)
10. [各数据源说明](#10-各数据源说明)
11. [LLM 排序与总结机制](#11-llm-排序与总结机制)
12. [常见问题与排查](#12-常见问题与排查)
13. [运行测试](#13-运行测试)
14. [许可证](#14-许可证)

---

## 1. 项目简介

**学术文献收集助手**是一个基于 Python 的 AI 工具，它将多个学术数据库的搜索能力与大语言模型（OpenAI GPT 或 Anthropic Claude）整合在一起，帮助研究人员：

- **一键式文献调研**：同时检索 Web of Science、谷歌学术、中国知网（CNKI）、IEEE Xplore 以及开放互联网，无需逐一登录不同平台。
- **自动去重与汇总**：将来自各数据源的结果合并，并按标题去除重复条目。
- **AI 智能排序**：使用大语言模型对检索结果按相关性打分（0–1），筛选出最有价值的文献。
- **结构化研究报告**：生成包含排名文献、关键主题、研究空白与叙述性摘要的完整研究报告。
- **MCP 协议支持**：通过 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) 将所有工具暴露给 Claude Desktop 或其他兼容客户端，实现"对话式"文献检索。

---

## 2. 功能特性

| 功能 | 说明 |
|---|---|
| **多数据源并发检索** | Web of Science · 谷歌学术 · 中国知网（CNKI）· IEEE Xplore · 通用网络搜索，全部并发执行 |
| **自动去重** | 基于标题归一化（大小写不敏感）自动合并重复文献 |
| **LLM 智能排序** | 调用 OpenAI、Anthropic 或 DeepSeek 对文献按相关性评分（0–1） |
| **研究报告生成** | 包含叙述性摘要、关键主题列表、研究空白分析 |
| **MCP 服务器** | 将 7 个搜索/排序工具通过 MCP 协议暴露给 Claude Desktop 等客户端 |
| **命令行界面（CLI）** | 直接在终端运行完整的文献调研会话 |
| **Python API** | 在自己的代码中以异步方式调用 `AcademicAssistant` 类 |

---

## 3. 系统架构

```
Academic-materials-collection-assistance/
├── main.py                            # 统一入口：CLI 或 MCP 服务器
├── requirements.txt                   # Python 依赖列表
├── pyproject.toml                     # 项目元数据与打包配置
├── .env.example                       # 环境变量示例文件
│
└── academic_assistant/                # 核心 Python 包
    ├── __init__.py
    ├── config.py                      # 配置管理（读取 .env 或环境变量）
    ├── assistant.py                   # 高层编排器 + CLI 入口
    │
    ├── models/
    │   └── paper.py                   # Pydantic 数据模型（Paper、SearchResult、ResearchReport 等）
    │
    ├── searchers/                     # 各数据源搜索适配器
    │   ├── base.py                    # 抽象基类 BaseSearcher
    │   ├── web_of_science.py          # Web of Science（Clarivate REST API）
    │   ├── google_scholar.py          # 谷歌学术（scholarly 库）
    │   ├── cnki.py                    # 中国知网（HTML 爬取）
    │   ├── ieee.py                    # IEEE Xplore（官方 REST API）
    │   └── web_search.py              # 通用网络搜索（Serper / Brave / DuckDuckGo）
    │
    ├── processors/
    │   └── ranker.py                  # LLM 驱动的排序与摘要生成
    │
    └── mcp_server/
        └── server.py                  # MCP 服务器（暴露 7 个工具）
```

### 数据流概览

```
用户查询
   │
   ▼
AcademicAssistant.research()
   │
   ├─► WebOfScienceSearcher  ─┐
   ├─► GoogleScholarSearcher  │  并发执行
   ├─► CNKISearcher           ├─► 合并 + 去重 ──► PaperRanker ──► ResearchReport
   ├─► IEEESearcher           │                     │
   └─► WebSearcher           ─┘              调用 LLM API
```

---

## 4. 环境要求

| 依赖项 | 最低版本 |
|---|---|
| Python | 3.11 |
| pip | 任意最新版本 |
| （可选）Git | 用于克隆仓库 |

**必须**至少拥有以下两类 API Key 之一才能使用 LLM 排序功能：

- **OpenAI API Key**（推荐，支持 `gpt-4o` 等模型）
- **Anthropic API Key**（支持 Claude 系列模型）

其他 API Key（WoS、IEEE、Serper、Brave）为可选项，不配置时对应数据源会返回错误，不影响其他数据源的正常检索。

---

## 5. 安装步骤

### 5.1 克隆仓库

```bash
git clone https://github.com/mintynibs-a11y/Academic-materials-collection-assistance.git
cd Academic-materials-collection-assistance
```

### 5.2 创建虚拟环境（推荐）

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 5.3 安装依赖

```bash
# 安装运行时依赖（不含开发/测试工具）
pip install -e .

# 或安装含测试工具的完整开发依赖
pip install -e ".[dev]"
```

> **说明**：`-e` 表示以"可编辑模式"安装，代码修改后无需重新安装即可生效。

---

## 6. 配置说明

所有配置通过环境变量传入，支持 `.env` 文件（推荐）或直接在 shell 中设置。

### 6.1 创建 `.env` 文件

```bash
cp .env.example .env
# 用文本编辑器打开 .env，填入你的 API Key
```

### 6.2 完整配置项说明

```dotenv
# =================== LLM 大语言模型 ===================

# 选择 LLM 提供商：openai（默认）、anthropic 或 deepseek
LLM_PROVIDER=openai

# OpenAI 配置
# 获取地址：https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...
# 使用的模型，默认 gpt-4o（支持 JSON 模式）
OPENAI_MODEL=gpt-4o

# Anthropic 配置（与 OpenAI / DeepSeek 三选一即可）
# 获取地址：https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-...
# 使用的模型，默认 claude-3-5-sonnet-20241022
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# DeepSeek 配置（与 OpenAI / Anthropic 三选一即可）
# 获取地址：https://platform.deepseek.com/
DEEPSEEK_API_KEY=sk-...
# 使用的模型，默认 deepseek-chat
DEEPSEEK_MODEL=deepseek-chat
# API 基础地址（使用官方服务无需修改）
DEEPSEEK_BASE_URL=https://api.deepseek.com

# =================== 学术数据库 API ===================

# Web of Science（Clarivate）
# 申请地址：https://developer.clarivate.com/
# 说明：无此 Key 时 WoS 检索将跳过，不影响其他数据源
WOS_API_KEY=

# IEEE Xplore
# 申请地址：https://developer.ieee.org/
# 说明：同上，无 Key 时 IEEE 检索跳过
IEEE_API_KEY=

# =================== 通用网络搜索 ===================

# Serper（Google 搜索 API，推荐）
# 申请地址：https://serper.dev/
SERPER_API_KEY=

# Brave Search API（备选）
# 申请地址：https://api.search.brave.com/
BRAVE_API_KEY=

# 说明：若两者均未配置，则自动回退到 DuckDuckGo HTML 抓取（无需 Key，
#        但结果质量和稳定性相对较低）

# =================== 中国知网（CNKI）===================

# 登录 cnki.net 后，从浏览器开发者工具中复制 Cookie 字符串
# 此项可选；不配置时仍可访问公开搜索结果，但结果可能受限
CNKI_SESSION_COOKIE=

# CNKI 请求代理（可选，格式：http://host:port）
CNKI_PROXY=

# =================== 搜索默认值 ===================

# 每个数据源默认返回的最大结果数
DEFAULT_MAX_RESULTS=10

# =================== MCP 服务器 ===================

# 传输方式：stdio（用于 Claude Desktop）或 sse（HTTP 模式）
MCP_TRANSPORT=stdio
# SSE 模式下监听的地址与端口
MCP_HOST=127.0.0.1
MCP_PORT=8765

# =================== 其他 ===================

# 自定义 HTTP 请求的 User-Agent 字符串（可选）
# ACADEMIC_ASSISTANT_USER_AGENT=Mozilla/5.0 ...
```

---

## 7. 快速上手

### 7.1 命令行（CLI）

#### 基础用法

```bash
python main.py "大语言模型在临床医学中的应用"
```

#### 完整参数说明

```bash
python main.py <研究主题> [--top-n N] [--max-per-source N]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `<研究主题>` | 必填 | 自然语言研究查询词，支持中英文 |
| `--top-n N` | 10 | 最终报告中展示的最相关文献数量 |
| `--max-per-source N` | 10 | 每个数据源最多返回的结果数量 |

#### 使用示例

```bash
# 中文查询
python main.py "Transformer 模型在蛋白质结构预测中的应用"

# 英文查询，展示 Top 20，每源最多 15 条
python main.py "deep learning protein folding" --top-n 20 --max-per-source 15

# 限制返回数量以加快速度
python main.py "量子计算综述" --top-n 5 --max-per-source 5
```

#### 输出示例

```
======================================================================
RESEARCH REPORT: Transformer 模型在蛋白质结构预测中的应用
======================================================================

Sources searched: ieee, google_scholar, cnki, web
Total papers found: 32
Papers ranked: 10

--- RANKED PAPERS ---

  #1  [0.97]  Highly Accurate Protein Structure Prediction with AlphaFold
       Authors : Jumper, J. et al.
       Year    : 2021
       Journal : Nature
       Citations: 28000
       DOI     : 10.1038/s41586-021-03819-2
       Reason  : 直接研究用于蛋白质结构预测的深度学习架构。

  #2  [0.93]  ...

--- KEY THEMES ---
  • 注意力机制在生物信息学中的应用
  • 预训练语言模型
  ...

--- RESEARCH GAPS / FUTURE DIRECTIONS ---
  • 多链蛋白质复合体预测的精度有待提升
  ...

--- SUMMARY ---
近年来，以 AlphaFold 为代表的基于 Transformer 的深度学习方法
在蛋白质结构预测领域取得了突破性进展...
======================================================================
```

---

### 7.2 MCP 服务器与 Claude Desktop 集成

**MCP（Model Context Protocol）** 允许将本工具的搜索与排序能力作为"工具"直接暴露给 Claude Desktop，在对话中随时调用。

#### 第一步：启动 MCP 服务器

```bash
# 通过统一入口以 stdio 模式启动（推荐，Claude Desktop 默认使用此模式）
python main.py --mcp-server
```

> 服务器启动后会在 stdout/stdin 上监听 MCP 消息，日志输出到 stderr。

#### 第二步：配置 Claude Desktop

找到 Claude Desktop 的配置文件（路径因操作系统而异）：

| 操作系统 | 配置文件路径 |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

向文件中的 `"mcpServers"` 对象添加如下配置：

```json
{
  "mcpServers": {
    "academic-assistant": {
      "command": "python",
      "args": [
        "/绝对路径/到/Academic-materials-collection-assistance/main.py",
        "--mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-你的Key",
        "WOS_API_KEY": "你的WoS Key（可选）",
        "IEEE_API_KEY": "你的IEEE Key（可选）",
        "SERPER_API_KEY": "你的Serper Key（可选）"
      }
    }
  }
}
```

> **注意**：`args` 中的路径必须是**绝对路径**。如果你使用了虚拟环境，将 `"command"` 改为虚拟环境中 Python 解释器的完整路径，例如：
> ```json
> "command": "/绝对路径/到/.venv/bin/python"
> ```

#### 第三步：重启 Claude Desktop

重启后，在 Claude 对话中即可使用以下工具（Claude 会自动在适当时机调用）：

```
你：帮我搜索关于"量子机器学习"的最新论文，并整理一份研究报告。

Claude：好的，我来使用 search_all_sources 工具同时检索所有数据库...
（自动调用工具，返回结果后再调用 rank_and_summarize）
```

---

### 7.3 Python 编程接口

在自己的 Python 项目中异步调用 `AcademicAssistant`：

#### 基础用法

```python
import asyncio
from academic_assistant.assistant import AcademicAssistant

async def main():
    # 创建助手实例（每个数据源最多返回 10 条结果）
    assistant = AcademicAssistant(max_results_per_source=10)

    # 执行完整研究会话，返回 Top 15 相关文献
    report = await assistant.research(
        "深度学习在医学影像诊断中的应用",
        top_n=15,
    )

    # 打印格式化报告
    AcademicAssistant.print_report(report)

    # 或直接访问报告字段
    print(f"共找到 {report.total_papers_found} 篇文献")
    print(f"关键主题：{report.key_themes}")
    for rp in report.ranked_papers:
        print(f"#{rp.rank} [{rp.relevance_score:.2f}] {rp.paper.title}")

asyncio.run(main())
```

#### 只搜索特定数据源

```python
from academic_assistant.assistant import AcademicAssistant
from academic_assistant.models.paper import Source

async def main():
    assistant = AcademicAssistant(
        max_results_per_source=20,
        sources=[Source.IEEE, Source.WEB_OF_SCIENCE],  # 只检索 IEEE 和 WoS
    )
    report = await assistant.research("neural architecture search")
    AcademicAssistant.print_report(report)
```

#### 直接使用单个搜索器

```python
import asyncio
from academic_assistant.searchers.ieee import IEEESearcher
from academic_assistant.config import config

async def main():
    searcher = IEEESearcher(api_key=config.IEEE_API_KEY)
    result = await searcher.search("federated learning", max_results=20)

    if result.success:
        for paper in result.papers:
            print(f"{paper.title} ({paper.year}) - {paper.formatted_authors}")
    else:
        print(f"检索失败：{result.error}")

asyncio.run(main())
```

#### 将报告序列化为 JSON

```python
import json
# report 是 ResearchReport 实例
json_str = report.model_dump_json(indent=2, ensure_ascii=False)
with open("report.json", "w", encoding="utf-8") as f:
    f.write(json_str)
```

---

## 8. MCP 工具参考

以 MCP 服务器模式运行时，以下 7 个工具可供调用：

### `search_web_of_science`

在 Web of Science（Clarivate）中检索学术论文，结果按被引次数降序排列。

**需要**：`WOS_API_KEY` 环境变量

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词 |
| `max_results` | integer | ❌ | 10 | 最多返回结果数（上限 50） |

---

### `search_google_scholar`

检索谷歌学术，无需 API Key。谷歌可能对自动化请求进行限速，频繁调用建议配置代理。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词（支持中英文） |
| `max_results` | integer | ❌ | 10 | 最多返回结果数 |

---

### `search_cnki`

检索中国知网（CNKI），无需 API Key，但配置 `CNKI_SESSION_COOKIE` 可获得更好的访问效果。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词（中英文均可） |
| `max_results` | integer | ❌ | 10 | 最多返回结果数 |

---

### `search_ieee`

检索 IEEE Xplore，覆盖电气、电子、计算机科学领域论文。

**需要**：`IEEE_API_KEY` 环境变量

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词 |
| `max_results` | integer | ❌ | 10 | 最多返回结果数（上限 200） |

---

### `search_web`

在互联网上进行通用学术文献搜索。

优先使用 `SERPER_API_KEY`（Google 搜索质量最佳），其次 `BRAVE_API_KEY`，最后回退到 DuckDuckGo HTML 抓取（无需 Key）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词 |
| `max_results` | integer | ❌ | 10 | 最多返回结果数 |

---

### `search_all_sources`

同时向所有 5 个数据源发起检索，自动聚合并去重。适合进行全面文献综述。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 检索词 |
| `max_results_per_source` | integer | ❌ | 5 | 每个数据源返回的最大结果数 |

**返回格式**（JSON）：

```json
{
  "total_papers": 23,
  "errors": ["web_of_science: WOS_API_KEY not set"],
  "papers": [
    {
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "source": "ieee",
      ...
    }
  ]
}
```

---

### `rank_and_summarize`

接收一批文献（JSON 数组），调用 LLM 按与查询词的相关性排序，并生成结构化研究报告。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 原始研究查询词（用于上下文） |
| `papers_json` | string | ✅ | — | JSON 格式的文献数组（来自搜索工具的返回值） |
| `top_n` | integer | ❌ | 10 | 报告中展示的 Top N 相关文献数 |

**返回**：`ResearchReport` 对象的 JSON 序列化，包含 `ranked_papers`、`summary`、`key_themes`、`research_gaps`。

---

## 9. 数据模型说明

所有数据模型基于 **Pydantic v2** 实现，位于 `academic_assistant/models/paper.py`。

### `Paper` — 单篇文献

| 字段 | 类型 | 说明 |
|---|---|---|
| `title` | str | 论文标题（必填） |
| `authors` | list[str] | 作者列表 |
| `abstract` | str \| None | 摘要 |
| `year` | int \| None | 发表年份（≥ 1000） |
| `published_date` | date \| None | 完整发表日期 |
| `journal` | str \| None | 期刊或会议名称 |
| `doi` | str \| None | DOI 编号 |
| `url` | str \| None | 文献链接 |
| `pdf_url` | str \| None | PDF 直链（如有） |
| `citations` | int \| None | 被引次数 |
| `keywords` | list[str] | 关键词列表 |
| `source` | Source | 数据来源（枚举） |
| `relevance_score` | float \| None | LLM 分配的相关性分数（0–1） |

**常用属性**：

```python
paper.formatted_authors  # "张三, 李四, 王五" 或 "张三 et al."
paper.short_summary()    # "标题 (2024) — 张三 et al., cited 100×"
```

### `Source` — 数据来源枚举

| 枚举值 | 说明 |
|---|---|
| `web_of_science` | Web of Science |
| `google_scholar` | 谷歌学术 |
| `cnki` | 中国知网 |
| `ieee` | IEEE Xplore |
| `arxiv` | arXiv（预留） |
| `web` | 通用网络搜索 |
| `unknown` | 未知来源 |

### `SearchResult` — 单个数据源的检索结果

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | str | 原始查询词 |
| `source` | Source | 数据来源 |
| `papers` | list[Paper] | 检索到的文献列表 |
| `total_found` | int \| None | 数据源报告的总结果数 |
| `error` | str \| None | 错误信息（若检索失败） |
| `success` | bool（属性） | 是否成功（`error is None`） |

### `RankedPaper` — 已排序的文献

| 字段 | 类型 | 说明 |
|---|---|---|
| `paper` | Paper | 文献对象 |
| `rank` | int | 排名（从 1 开始） |
| `relevance_score` | float | 相关性分数（0–1） |
| `reason` | str | LLM 给出的排名理由 |

### `ResearchReport` — 完整研究报告

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | str | 研究查询词 |
| `sources_searched` | list[Source] | 检索过的数据源列表 |
| `total_papers_found` | int | 去重后总文献数 |
| `ranked_papers` | list[RankedPaper] | 排序后的 Top N 文献 |
| `summary` | str | LLM 生成的叙述性总结 |
| `key_themes` | list[str] | 主要研究主题 |
| `research_gaps` | list[str] | 研究空白与未来方向 |

---

## 10. 各数据源说明

### Web of Science

- **接口**：Clarivate WoS Starter API v1（REST）
- **API Key 申请**：[developer.clarivate.com](https://developer.clarivate.com/)
- **覆盖范围**：Science Citation Index、Social Sciences Citation Index 等核心期刊数据库
- **排序方式**：按被引次数降序
- **每次请求上限**：50 条

### 谷歌学术

- **接口**：[scholarly](https://github.com/scholarly-python-package/scholarly) 开源库（网页抓取）
- **无需 API Key**
- **注意**：谷歌可能封锁高频爬取请求。如遇封锁，可通过 `SCHOLARLY_PROXY` 环境变量配置代理（如 Tor）：
  ```dotenv
  SCHOLARLY_PROXY=socks5://127.0.0.1:9050
  ```

### 中国知网（CNKI）

- **接口**：对 `kns.cnki.net` 进行 HTML 抓取（无官方公开 API）
- **无需 API Key**（可选 `CNKI_SESSION_COOKIE` 提升访问效果）
- **获取 Cookie**：在浏览器中登录 cnki.net → 按 F12 打开开发者工具 → Network → 找到任意请求 → 复制 Cookie 请求头的值，粘贴到 `.env` 中
- **代理支持**：可通过 `CNKI_PROXY` 设置请求代理

### IEEE Xplore

- **接口**：IEEE Xplore REST API v1（官方）
- **API Key 申请**：[developer.ieee.org](https://developer.ieee.org/)（免费注册）
- **覆盖范围**：IEEE 学报、会议论文集、标准文档
- **每次请求上限**：200 条

### 通用网络搜索

按优先级依次尝试以下后端：

1. **Serper**（推荐）：通过 Google 搜索 API 获取高质量结果，需要 `SERPER_API_KEY`，每月 2500 次免费额度
2. **Brave Search**：隐私友好的搜索引擎 API，需要 `BRAVE_API_KEY`
3. **DuckDuckGo**（自动回退）：无需 API Key，直接抓取 HTML 搜索结果页，可能受限速影响

---

## 11. LLM 排序与总结机制

`PaperRanker`（位于 `academic_assistant/processors/ranker.py`）的工作流程：

1. **压缩文献信息**：将每篇文献的标题、作者、年份、期刊、摘要（截断至 500 字符）、被引次数、关键词整理为紧凑 JSON
2. **构造 Prompt**：将研究查询词、压缩文献列表发送给 LLM（系统提示要求以专家学术分析师身份响应）
3. **解析响应**：LLM 以严格 JSON 格式返回排名列表（含相关性分数和理由）、叙述性摘要、主题列表、研究空白列表
4. **构建报告**：将排名结果与原始文献对象关联，组装为 `ResearchReport`

### 切换 LLM 提供商

```dotenv
# 使用 OpenAI（默认）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# 或使用 Anthropic Claude
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# 或使用 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
```

三者只需设置其中一个即可，`LLM_PROVIDER` 控制实际调用哪个。DeepSeek 使用与 OpenAI 兼容的 API 接口，国内访问速度通常更快且费用较低。

---

## 12. 常见问题与排查

### Q: 运行后看不到任何文献结果

**可能原因 1**：所有数据源均未配置 API Key，且网络无法访问谷歌学术 / DuckDuckGo。

**解决**：至少配置一个可用数据源（推荐先配置 `SERPER_API_KEY` 或 `IEEE_API_KEY`），并确保网络连通。

---

### Q: LLM 返回空的排序结果

**可能原因**：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY` 未配置或无效。

**解决**：检查 `.env` 文件，确认所选提供商的 Key 填写正确且账户有余额/权限。日志中搜索 `"not set"` 或 `"call failed"` 可确认具体原因。如果网络访问 OpenAI/Anthropic 受限，可切换为 `LLM_PROVIDER=deepseek` 并配置 `DEEPSEEK_API_KEY`。

---

### Q: 谷歌学术检索报错或返回为空

**可能原因**：谷歌对自动化请求进行了限速或封锁（出现 CAPTCHA）。

**解决**：
1. 降低 `--max-per-source` 参数值，减少单次请求量
2. 配置代理：在 `.env` 中设置 `SCHOLARLY_PROXY=socks5://127.0.0.1:9050`（需要安装并运行 Tor）
3. 临时禁用谷歌学术，仅检索其他数据源

---

### Q: CNKI 检索结果为空或解析失败

**可能原因**：CNKI 页面结构可能发生变化，或需要登录态才能看到完整结果。

**解决**：
1. 登录 cnki.net，从浏览器复制 Cookie 到 `CNKI_SESSION_COOKIE`
2. 如果在中国大陆以外访问，可能需要配置 `CNKI_PROXY`

---

### Q: 如何在没有任何 API Key 的情况下试用？

可以仅依赖 DuckDuckGo 回退搜索和谷歌学术（scholarly 库），LLM 功能需要至少一个 LLM API Key。最低配置：

```dotenv
OPENAI_API_KEY=sk-...  # 必须
# 其他 Key 全部留空，谷歌学术 + DuckDuckGo 无需 Key
```

---

### Q: MCP 服务器启动后 Claude Desktop 看不到工具

**排查步骤**：
1. 确认 `claude_desktop_config.json` 中路径为**绝对路径**
2. 确认使用的 Python 路径正确（如使用虚拟环境，路径应为 `.venv/bin/python`）
3. 重启 Claude Desktop 后等待约 10 秒
4. 在 Claude 对话中输入"你有哪些工具可以使用？"确认工具是否出现

---

## 13. 运行测试

测试使用 `pytest` 框架，所有测试均位于 `tests/` 目录下，不依赖真实 API（通过 mock 模拟网络请求）。

```bash
# 安装开发依赖（如尚未安装）
pip install -e ".[dev]"

# 运行所有测试
pytest tests/ -v

# 只运行某个测试文件
pytest tests/test_models.py -v
pytest tests/test_searchers.py -v
pytest tests/test_ranker.py -v
pytest tests/test_assistant.py -v
```

测试覆盖范围：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_models.py` | `Paper`、`SearchResult`、`ResearchReport` 的字段验证与序列化 |
| `test_searchers.py` | 各搜索器的错误处理、HTML/JSON 解析逻辑 |
| `test_ranker.py` | LLM 排序流程（mock LLM 调用）、空输入处理 |
| `test_assistant.py` | 去重逻辑、并发搜索编排、报告打印 |

---

## 14. 许可证

MIT License — 详见项目根目录（如有 LICENSE 文件）。
