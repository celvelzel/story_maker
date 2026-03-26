# KG 开关工况对比报告

- 生成时间: 2026-03-26 16:33:25
- Genre: fantasy
- 每工况运行次数: 1
- 每次最大轮次: 5
- 动作策略: 每回合固定选择第一个选项

## 1. 会话明细

| Session | Condition | Genre | Turns | Mean(s) | P50(s) | P90(s) | P95(s) | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | kg_on | fantasy | 5 | 20.0308 | 19.7709 | 24.7825 | 26.0786 | OK |
| 2 | kg_off | fantasy | 5 | 5.1654 | 4.6301 | 6.4566 | 6.9737 | OK |

## 2. 延迟指标对比（kg_on - kg_off）

| Metric | kg_on | kg_off | Delta |
|---|---:|---:|---:|
| response_mean_s | 20.0308 | 5.1654 | 14.8654 |
| response_p50_s | 20.0308 | 5.1654 | 14.8654 |
| response_p90_s | 20.0308 | 5.1654 | 14.8654 |
| response_p95_s | 20.0308 | 5.1654 | 14.8654 |
| response_std_s | 0.0000 | 0.0000 | 0.0000 |
| response_min_s | 20.0308 | 5.1654 | 14.8654 |
| response_max_s | 20.0308 | 5.1654 | 14.8654 |

## 3. 自动指标对比（metrics.py）

| Metric | kg_on | kg_off | Delta |
|---|---:|---:|---:|
| distinct_1 | 0.4417 | 0.4123 | 0.0294 |
| distinct_2 | 0.8283 | 0.7742 | 0.0541 |
| distinct_3 | 0.9574 | 0.8999 | 0.0575 |
| self_bleu | 0.1271 | 0.2514 | -0.1243 |
| entity_coverage | 0.6250 | 1.0000 | -0.3750 |
| consistency_rate | 1.0000 | 1.0000 | 0.0000 |
| type_token_ratio | 0.3938 | 0.3760 | 0.0179 |
| flesch_reading_ease | 53.9900 | 50.9700 | 3.0200 |
| lexical_overlap | 0.2764 | 0.2976 | -0.0212 |
| graph_density_average | 0.0814 | 0.0000 | 0.0814 |
| graph_density_delta | 0.0893 | 0.0000 | 0.0893 |

## 4. LLM Judge 对比（llm_judge.py）

| Dimension | kg_on | kg_off | Delta |
|---|---:|---:|---:|
| narrative_quality | 9.00 | 9.00 | 0.00 |
| consistency | 10.00 | 10.00 | 0.00 |
| player_agency | 8.00 | 8.00 | 0.00 |
| creativity | 8.00 | 7.00 | 1.00 |
| pacing | 9.00 | 9.00 | 0.00 |
| option_relevance | 9.00 | 9.00 | 0.00 |
| causal_link | 9.00 | 9.00 | 0.00 |
| local_coherence | 10.00 | 10.00 | 0.00 |
| average | 9.00 | 8.88 | 0.12 |

## 5. 说明

- `Delta` 统一定义为 `kg_on - kg_off`。
- 本次为单次对比（runs=1），结果会受模型随机性与 API 抖动影响。
- 若评测模型不可用，LLM Judge 维度会回退到 0。
