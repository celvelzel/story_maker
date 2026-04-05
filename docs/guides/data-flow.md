# StoryWeaver 模块间数据传递（字段级）

> **Last Updated**: 2026-04-01

## 1. 单轮数据流总览

每轮 `process_turn(player_input)` 的数据流如下：

1. 输入层：玩家输入 `player_input`
2. NLU 层：得到 `resolved_input`、`intent`、`confidence`、`entities`
3. NLG 层：根据 NLU + `kg_summary` + `history` 生成 `story_text`
4. KG 层：更新实体关系，输出 `kg_summary`、`kg_html`、`conflicts`
5. UI 层：渲染故事文本、分支选项、NLU debug、KG 图

## 2. 核心结构

### 2.1 TurnResult

字段说明：

1. `story_text: str`：本轮叙事输出
2. `options: List[StoryOption]`：可选行动
3. `nlu_debug: Dict`：NLU 调试信息
4. `kg_html: str`：图谱可视化 HTML
5. `conflicts: List[str]`：一致性冲突文本

### 2.2 nlu_debug

常见字段：

1. `resolved_input`
2. `intent`
3. `confidence`
4. `entities`
5. `intent_backend`
6. `intent_model_loaded`
7. `coref_loaded`
8. `entity_model_loaded`

### 2.3 kg_summary

来源：`KnowledgeGraph.to_summary()`

用途：

1. 为 story_generator 提供世界状态摘要
2. 为 option_generator 提供上下文约束

### 2.4 history

来源：`GameState.recent_history(window)`

用途：

1. 提供近轮上下文
2. 保证叙事连贯与角色一致性

## 3. 逐 turn 字段流转

1. 输入：`player_input`
2. 经过 coref：`resolved_input`
3. 意图分类：`intent`, `confidence`
4. 实体识别：`entities`
5. 主状态写入：`state.story_history` 追加 player
6. 文本生成：`story_text`
7. 状态写入：`state.story_history` 追加 narrator
8. KG 增量：
   - `entities` 先写入
   - `story_text` 抽取 relations 再写入
9. 冲突检测：`conflicts`
10. 选项生成：`options`
11. 结果封装：`TurnResult`

## 4. NLG 后端选择

项目支持三种 NLG 后端（通过 `config.py` 的 `NLG_MODE` 控制）：

| 模式 | 说明 |
|------|------|
| `api` | 远程 OpenAI 兼容 API（默认 Mimo API） |
| `local` | 本地 llama.cpp 服务器（通过 `OPENAI_BASE_URL` 指向 127.0.0.1:8081） |
| `hybrid` | 混合模式：创意任务用本地 Qwen3，结构化任务用远程 API |

### 关键配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `NLG_MODE` | `"hybrid"` | NLG 后端选择 |
| `OPENAI_BASE_URL` | 远程 API URL | 本地模式改为 `http://127.0.0.1:8081/v1` |
| `OPENAI_MODEL` | `"mimo-v2-flash"` | 本地模式改为 `"qwen3-4b"` |
| `OPENAI_TIMEOUT_CONNECT` | `10.0` | 连接超时（秒），CPU 推理建议 `30.0` |
| `OPENAI_TIMEOUT_READ` | `60.0` | 读取超时（秒），CPU 推理建议 `180.0` |

## 5. 实现索引（关键文件）

1. `src/engine/game_engine.py` — 游戏主引擎
2. `src/engine/state.py` — 游戏状态管理
3. `src/nlu/intent_classifier.py` — 意图分类（DistilBERT + 关键词兜底）
4. `src/nlu/coreference.py` — 共指消解（fastcoref + 规则兜底）
5. `src/nlu/entity_extractor.py` — 实体抽取（spaCy NER + 启发式兜底）
6. `src/knowledge_graph/graph.py` — 知识图谱（NetworkX MultiDiGraph）
7. `src/knowledge_graph/relation_extractor.py` — 关系抽取（LLM 结构化输出）
8. `src/knowledge_graph/conflict_detector.py` — 冲突检测（规则 + LLM 校验）
9. `src/nlg/story_generator.py` — 故事生成（支持 api/local/hybrid）
10. `src/nlg/option_generator.py` — 选项生成
11. `config.py` — 全局配置（Pydantic Settings）
12. `app.py` — Streamlit 应用入口
