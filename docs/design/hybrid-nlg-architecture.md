# 混合 NLG 架构

> **最后更新：** 2026-04-01  
> **模块：** `src/nlg/`、`src/utils/api_client.py`

## 1. 概述

StoryWeaver 的 NLG（自然语言生成）模块采用**混合架构**，将不同类型的任务路由到不同的 LLM 后端，以同时优化质量与成本。系统支持三种运行模式：`api`、`local` 和 `hybrid`。

## 2. 架构

```
┌─────────────────────────────────────────────────────┐
│                NLG 请求                               │
│  （故事生成 / 选项生成 / KG 关系抽取）                  │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  NLG_MODE 检查   │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  "api"  │   │ "local" │   │"hybrid" │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        │              │    ┌────────┴────────┐
        │              │    │   任务路由       │
        │              │    └────────┬────────┘
        │              │             │
        │              │    ┌────────┴────────┐
        │              │    │ 故事 → 本地     │
        │              │    │ 选项 → API      │
        │              │    │ 关系 → API      │
        │              │    │ JSON → API      │
        ▼              ▼    ▼                 ▼
   ┌──────────────────────────────────────────────┐
   │              LLMClient（单例）                  │
   │  - 按类型的实例：LLMClient("api")              │
   │  - 按类型的实例：LLMClient("local")            │
   │  - chat() / chat_json() / 使用量追踪           │
   └──────────────────────────────────────────────┘
        │                                │
        ▼                                ▼
   ┌─────────────┐              ┌─────────────┐
   │  Mimo API   │              │ 本地 Qwen3  │
   │ （结构化任务）│              │ （创意任务）  │
   └─────────────┘              └─────────────┘
```

## 3. 路由逻辑

`src/utils/api_client.py` 中的 `HybridClientManager` 实现基于任务的路由：

| NLG_MODE | 故事生成 | 选项生成 | KG 关系抽取 | JSON 任务 |
|----------|-----------------|-------------------|----------------------|------------|
| `api` | Mimo/OpenAI API | Mimo/OpenAI API | Mimo/OpenAI API | Mimo/OpenAI API |
| `local` | 本地 Qwen3 | 本地 Qwen3 | 本地 Qwen3 | 本地 Qwen3 |
| `hybrid` | **本地 Qwen3** | **Mimo API** | **Mimo API** | **Mimo API** |

### 混合模式的设计理由

- **创意任务（故事）**：受益于本地模型的一致行为表现，且每 token 无 API 成本。
- **结构化任务（选项、关系、JSON）**：受益于云端 API 更优秀的 JSON 合规性和指令遵循能力。

## 4. LLMClient 设计

### 4.1 按类型的单例模式

`LLMClient` 使用按类型的单例模式：

```python
# 每种类型获得独立的单例实例
api_client = LLMClient(client_type="api")    # 使用 MIMO_* 或 OPENAI_* 设置
local_client = LLMClient(client_type="local") # 使用 OPENAI_BASE_URL
```

### 4.2 客户端配置

| 客户端类型 | API 密钥 | 基础 URL | 模型 |
|-------------|---------|----------|-------|
| `local`（本地） | `OPENAI_API_KEY`（或 "not-needed"） | `OPENAI_BASE_URL` | `OPENAI_MODEL` |
| `api`（云端） | `MIMO_API_KEY`（回退：`OPENAI_API_KEY`） | `MIMO_BASE_URL`（回退：`OPENAI_BASE_URL`） | `OPENAI_MODEL` |

### 4.3 核心方法

- `chat(messages, temperature, max_tokens, json_mode)` — 文本补全，支持重试（3 次尝试，指数退避）
- `chat_json(messages, temperature, max_tokens)` — JSON 补全，含多阶段修复：
  1. Markdown 围栏剥离
  2. 平衡 JSON 对象提取
  3. 尾逗号移除
  4. 严格重试（temperature=0.0）
- `usage_snapshot()` / `usage_delta(before, after)` — 每阶段 token 使用量追踪

## 5. 错误处理

- **重试机制：** 最多 3 次尝试，退避间隔 1s、2s、3s
- **JSON 修复：** 多候选解析，含围栏剥离、平衡提取、尾逗号移除和严格重试
- **兜底方案：** 选项生成回退到硬编码选项；KG 抽取返回空结果

## 6. 配置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `NLG_MODE` | `hybrid` | 路由模式：`api`、`local` 或 `hybrid` |
| `OPENAI_API_KEY` | `""` | 本地端点的 API 密钥 |
| `OPENAI_BASE_URL` | `""` | 本地 llama.cpp 服务器 URL |
| `OPENAI_MODEL` | `mimo-v2-flash` | 模型名称 |
| `OPENAI_TEMPERATURE` | `0.85` | 默认生成温度 |
| `OPENAI_MAX_TOKENS` | `1024` | 默认最大 token 数 |
| `OPENAI_TIMEOUT_CONNECT` | `10.0` | 连接超时（秒） |
| `OPENAI_TIMEOUT_READ` | `60.0` | 读取超时（秒） |

## 7. 使用方式

```python
from src.utils.api_client import llm_client, get_client_for_task

# 直接使用（遵循 NLG_MODE）
response = llm_client.chat([
    {"role": "system", "content": "You are a narrator."},  # 系统提示：你是叙述者
    {"role": "user", "content": "Describe a dark forest."},  # 用户请求：描述一片黑暗森林
])

# 基于任务的路由（用于混合模式）
story_client = get_client_for_task("story")      # 混合模式下 → 本地
option_client = get_client_for_task("option")    # 混合模式下 → API
```

---
*相关文档：本地模型训练详情请见 [nlg-local-model-finetuning.md](nlg-local-model-finetuning.md)*
