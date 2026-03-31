# StoryWeaver 自动化测试报告

- 生成时间: 2026-03-26 16:09:19
- 测试目标: 评估系统自动指标、LLM 主观评分与端到端响应时间

## 1. 测试配置

- 会话总数: 3
- Genre 列表: fantasy, sci-fi, mystery
- 每会话最大轮次 (max_turns): 10
- 预热轮次 (warmup): 0
- 动作策略: 每回合自动选择 branch options 的第一个选项

## 2. 会话明细

| Session | Genre | Turns | Mean(s) | P50(s) | P90(s) | P95(s) | Std(s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | fantasy | 10 | 18.9360 | 18.7056 | 22.2735 | 23.3086 | 3.0632 | OK |
| 2 | sci-fi | 10 | 19.6930 | 19.2864 | 23.3718 | 24.1221 | 3.0941 | OK |
| 3 | mystery | 10 | 20.6482 | 20.3414 | 22.7168 | 22.8156 | 1.4706 | OK |

## 3. 总体汇总（不按 Genre 分组）

### 3.1 响应时间

| Metric | Value (s) |
|---|---:|
| Mean | 19.7590 |
| P50 | 19.6930 |
| P90 | 20.4571 |
| P95 | 20.5527 |
| Std | 0.8580 |
| Min | 18.9360 |
| Max | 20.6482 |

### 3.2 自动指标（metrics.py）

| Metric | Value |
|---|---:|
| distinct_1 | 0.3493 |
| distinct_2 | 0.7083 |
| distinct_3 | 0.8921 |
| self_bleu | 0.2398 |
| entity_coverage | 0.9630 |
| consistency_rate | 1.0000 |
| type_token_ratio | 0.3060 |
| flesch_reading_ease | 53.5867 |
| lexical_overlap | 0.2817 |
| graph_density_average | 0.1084 |
| graph_density_delta | 0.0241 |

### 3.3 LLM Judge（llm_judge.py）

| Dimension | Value |
|---|---:|
| narrative_quality | 8.00 |
| consistency | 6.67 |
| player_agency | 6.67 |
| creativity | 6.33 |
| pacing | 8.00 |
| option_relevance | 7.33 |
| causal_link | 6.33 |
| local_coherence | 6.00 |
| average | 6.91 |

## 4. 参数解释与解读

- `max_turns=10`: 每会话最多 10 次用户动作，控制测试长度与成本。
- `warmup=0`: 不做预热，报告结果会包含冷启动影响。
- 动作策略固定首选项: 提高实验可复现性，减少随机动作导致的方差。
- `distinct_n`: 越高通常代表词汇/短语多样性更好。
- `self_bleu`: 越低通常代表重复度更低、生成更不模板化。
- `entity_coverage`: 越高表示生成文本覆盖更多知识图谱实体。
- `consistency_rate`: 越高表示每回合冲突更少、世界状态更稳定。
- `type_token_ratio`: 词汇丰富度（不同词占比），越高通常表示词汇变化更丰富。
- `flesch_reading_ease`: 可读性分数（越高越易读）。
- `lexical_overlap`: 相邻叙事文本词汇交集比例，用于衡量局部承接。
- `graph_density_average`: 会话期内 KG 平均密度，用于观察世界关系复杂度。
- `graph_density_delta`: 会话结束与开场的 KG 密度变化量。
- LLM Judge 八维评分（1-10）用于补充主观质量评估，`average` 为八维均值。
- 新增三维含义：`option_relevance`（选项相关性）、`causal_link`（因果链合理性）、`local_coherence`（相邻回合连贯性）。

## 5. 备注

- 错误会话数: 0。若 API 或模型加载失败，对应会话将记为 ERROR，并以 0 值填充指标。
