# Hybrid NLG Architecture

> **Last Updated:** 2026-04-01  
> **Module:** `src/nlg/`, `src/utils/api_client.py`

## 1. Overview

StoryWeaver's NLG (Natural Language Generation) module uses a **hybrid architecture** that routes different task types to different LLM backends, optimizing for both quality and cost. The system supports three operational modes: `api`, `local`, and `hybrid`.

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                NLG Request                          │
│  (Story Generation / Option Generation / KG Extract)│
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  NLG_MODE Check │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  "api"  │   │ "local" │   │"hybrid" │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        │              │    ┌────────┴────────┐
        │              │    │  Task Routing   │
        │              │    └────────┬────────┘
        │              │             │
        │              │    ┌────────┴────────┐
        │              │    │ story → local   │
        │              │    │ option → api    │
        │              │    │ relation → api  │
        │              │    │ json → api      │
        ▼              ▼    ▼                 ▼
   ┌──────────────────────────────────────────────┐
   │              LLMClient (Singleton)            │
   │  - Per-type instances: LLMClient("api")       │
   │  - Per-type instances: LLMClient("local")     │
   │  - chat() / chat_json() / usage tracking      │
   └──────────────────────────────────────────────┘
        │                                │
        ▼                                ▼
   ┌─────────────┐              ┌─────────────┐
   │  Mimo API   │              │ Local Qwen3 │
   │ (Structured)│              │  (Creative) │
   └─────────────┘              └─────────────┘
```

## 3. Routing Logic

The `HybridClientManager` in `src/utils/api_client.py` implements task-based routing:

| NLG_MODE | Story Generation | Option Generation | KG Relation Extraction | JSON Tasks |
|----------|-----------------|-------------------|----------------------|------------|
| `api` | Mimo/OpenAI API | Mimo/OpenAI API | Mimo/OpenAI API | Mimo/OpenAI API |
| `local` | Local Qwen3 | Local Qwen3 | Local Qwen3 | Local Qwen3 |
| `hybrid` | **Local Qwen3** | **Mimo API** | **Mimo API** | **Mimo API** |

### Rationale for Hybrid Mode

- **Creative tasks (story)** benefit from local model's consistent behavior and zero API cost per token.
- **Structured tasks (options, relations, JSON)** benefit from cloud API's superior JSON compliance and instruction following.

## 4. LLMClient Design

### 4.1 Multi-Type Singleton

The `LLMClient` uses a per-type singleton pattern:

```python
# Each type gets its own singleton instance
api_client = LLMClient(client_type="api")    # Uses MIMO_* or OPENAI_* settings
local_client = LLMClient(client_type="local") # Uses OPENAI_BASE_URL
```

### 4.2 Client Configuration

| Client Type | API Key | Base URL | Model |
|-------------|---------|----------|-------|
| `local` | `OPENAI_API_KEY` (or "not-needed") | `OPENAI_BASE_URL` | `OPENAI_MODEL` |
| `api` | `MIMO_API_KEY` (fallback: `OPENAI_API_KEY`) | `MIMO_BASE_URL` (fallback: `OPENAI_BASE_URL`) | `OPENAI_MODEL` |

### 4.3 Core Methods

- `chat(messages, temperature, max_tokens, json_mode)` — Text completion with retry (3 attempts, exponential backoff).
- `chat_json(messages, temperature, max_tokens)` — JSON completion with multi-stage repair:
  1. Markdown fence stripping
  2. Balanced JSON object extraction
  3. Trailing comma removal
  4. Strict retry with temperature=0.0
- `usage_snapshot()` / `usage_delta(before, after)` — Per-stage token usage tracking.

## 5. Error Handling

- **Retry:** Up to 3 attempts with 1s, 2s, 3s backoff.
- **JSON Repair:** Multi-candidate parsing with fence stripping, balanced extraction, trailing comma removal, and strict retry.
- **Fallback:** Option generation falls back to hardcoded options; KG extraction returns empty results.

## 6. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NLG_MODE` | `hybrid` | Routing mode: `api`, `local`, or `hybrid` |
| `OPENAI_API_KEY` | `""` | API key for local endpoint |
| `OPENAI_BASE_URL` | `""` | Local llama.cpp server URL |
| `OPENAI_MODEL` | `mimo-v2-flash` | Model name |
| `OPENAI_TEMPERATURE` | `0.85` | Default generation temperature |
| `OPENAI_MAX_TOKENS` | `1024` | Default max tokens |
| `OPENAI_TIMEOUT_CONNECT` | `10.0` | Connection timeout (seconds) |
| `OPENAI_TIMEOUT_READ` | `60.0` | Read timeout (seconds) |

## 7. Usage

```python
from src.utils.api_client import llm_client, get_client_for_task

# Direct usage (respects NLG_MODE)
response = llm_client.chat([
    {"role": "system", "content": "You are a narrator."},
    {"role": "user", "content": "Describe a dark forest."},
])

# Task-based routing (for hybrid mode)
story_client = get_client_for_task("story")      # → local in hybrid mode
option_client = get_client_for_task("option")    # → API in hybrid mode
```

---
*Related: [nlg-local-model-finetuning.md](nlg-local-model-finetuning.md) for local model training details.*
