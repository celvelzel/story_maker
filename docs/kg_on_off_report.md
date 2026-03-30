# KG 开关工况对比报告

- 生成时间: 2026-03-27 16:16:49
- Genre: fantasy
- 每工况运行次数: 1
- 每次最大轮次: 10
- 动作策略: 每回合固定选择第一个选项

## 1. 会话明细

| Session | Condition | Genre | Turns | Status |
|---|---|---|---:|---|
| 1 | kg_on | fantasy | 10 | OK |
| 2 | kg_off | fantasy | 10 | OK |

## 2. 自动指标对比（metrics.py）

| Metric | kg_on | kg_off |
|---|---:|---:|
| distinct_1 | 0.3652 | 0.3165 |
| distinct_2 | 0.7423 | 0.6448 |
| distinct_3 | 0.9059 | 0.8145 |
| self_bleu | 0.2200 | 0.3582 |
| entity_coverage | 1.0000 | 0.7917 |
| consistency_rate | 1.0000 | 0.0000 |
| type_token_ratio | 0.3137 | 0.2690 |
| flesch_reading_ease | 54.8500 | 61.1600 |
| lexical_overlap | 0.2745 | 0.3801 |
| graph_density_average | 0.0000 | 0.1575 |
| graph_density_delta | 0.0000 | -0.2123 |

## 3. LLM Judge 对比（llm_judge.py）

| Dimension | kg_on | kg_off |
|---|---:|---:|
| narrative_quality | 9.00 | 8.00 |
| consistency | 10.00 | 9.00 |
| player_agency | 8.00 | 6.00 |
| creativity | 8.00 | 6.00 |
| pacing | 9.00 | 7.00 |
| option_relevance | 9.00 | 7.00 |
| causal_link | 10.00 | 8.00 |
| local_coherence | 10.00 | 9.00 |
| average | 9.12 | 7.50 |

## 4. 结论分析

- LLM Judge（average，越高越好）：kg_on 更优 (kg_on=9.12, kg_off=7.50)。
- 自动指标投票（distinct_1/2/3, consistency_rate, type_token_ratio, self_bleu）：kg_on 胜 6 项, kg_off 胜 0 项，结论=kg_on。
- 综合判断：kg_on 更好（按 LLM Judge 与自动指标两类多数票）。

## 5. 说明

- 本次为单次对比（runs=1），结果会受模型随机性与 API 抖动影响。
- 若评测模型不可用，LLM Judge 维度会回退到 0。
