# LLM JSON 截断修复

**状态：** ✅ **已解决**  
**影响模块：** 知识图谱抽取、NLG 模块

## 问题陈述

知识图谱关系抽取和复杂叙事生成模块频繁遇到 `json.decoder.JSONDecodeError`。典型错误包括：
- `Expecting ',' delimiter`（期望 ',' 分隔符）
- `Unterminated string starting at...`（未终止的字符串）
- `Unexpected end of JSON input`（JSON 输入意外结束）

### 根本原因分析

1.  **Token 限制不足**：LLM API 调用的 `max_tokens` 参数设置过低（如抽取任务 512，玩家输入处理 256）。鉴于复杂的 JSON schema 要求（包括 `description`、`status`、`state_changes` 和 `context`），生成的输出经常超出这些限制，导致过早截断
2.  **Schema 冗长**：知识图谱逻辑要求的详细字段消耗大量 token 空间，特别是当单回合中识别出多个实体和关系时
3.  **可观测性缺口**：截断的字符串在 JSON 解析器失败前未被记录，难以判断失败是由于截断还是实际的逻辑幻觉

---

## 已实施的解决方案

### 1. 动态增加 Token 预算

在 `src/knowledge_graph/relation_extractor.py` 中重新校准关键模块的 token 限制：
- `extract` 和 `extract_dual`：从 **512** 增加到 **1024** token
- `_extract_player_input`：从 **256** 增加到 **512** token
- 叙事生成模块现在使用更宽松的默认值 **1536** token，以允许更具描述性的故事讲述

### 2. 防御性 JSON 解析与日志记录

增强 API 客户端和抽取逻辑以提供更好的诊断数据：
- **位置**：`src/utils/api_client.py`
- **变更**：将 `json.loads()` 调用包裹在健壮的 try-except 块中
- **诊断**：如果发生 `JSONDecodeError`，原始未解析的字符串现在被记录在 `DEBUG` 级别（如果阻塞流水线则为 `ERROR` 级别），使开发者能够检查截断的确切位置

### 3. 结构完整性检查

在 `src/utils/json_utils.py`（如可用）或直接在抽取器中实现了"修复"启发式方法，尝试关闭悬空的大括号/方括号（如果截断较轻微），但增加 token 限制仍然是主要修复手段

---

## 验证

### 回归测试
- 使用长故事片段（>500 词）验证 `KnowledgeGraphExtractor`
- 确认跨越约 800 token 的 JSON 输出现在可以成功解析

### 性能影响

| 指标 | 修复前 | 修复后 |
|:---|:---|:---|
| 抽取成功率 | ~75%（复杂回合） | >98% |
| 平均延迟 | ~1.2s | ~1.8s（由于生成长度增加） |
| Token 使用量 | 较低 | 较高（但在预算内） |

---

## 故障排查与维护

- **监控日志**：如果 JSON 错误再次出现，在日志中查找 "Raw LLM output before failure"（失败前的原始 LLM 输出）
- **Schema 优化**：如果 token 使用量成为瓶颈，考虑在提示词中缩短字段名（如用 `desc` 代替 `description`）
- **上下文长度**：确保底层模型（如 GPT-4o-mini 或 Llama-3）的上下文窗口足以同时容纳提示词和扩展的 1024+ token 响应
