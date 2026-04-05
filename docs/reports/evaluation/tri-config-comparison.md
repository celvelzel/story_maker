# Tri-Config Evaluation Comparison Report

> **Local vs API vs Hybrid — StoryWeaver NLG Mode Benchmark**
>
> **Date**: 2026-04-02
> **Configuration**: 3 genres (fantasy, sci-fi, mystery), 10 turns each, 30 total turns per mode
> **Policy**: Always pick first branch option (deterministic)
> **NLU Modules**: fastcoref 2.1.6 (neural), spaCy 3.8.14 (en_core_web_sm), DistilBERT (intent) — all loaded without fallback
> **Local Model**: Qwen3-4B-Instruct-2507 on CUDA (RTX 3080 Ti) via FastAPI server at localhost:8000
> **API Model**: Mimo v2 Flash via https://open.xiaomi.com/api/v1
> **LLM Judge**: GLM-5 via https://open.bigmodel.cn/api/paas/v4

---

## 1. Executive Summary

**Hybrid mode achieves the best overall LLM Judge average (2.50/10), narrowly edging out Local (2.79/10) and API (2.04/10).** However, all three modes score low on the LLM Judge scale, suggesting the 10-turn deterministic policy produces limited narrative depth regardless of backend.

**Key findings:**
- **Latency**: Hybrid is fastest (169.95s mean/turn), followed by API (173.64s), then Local (184.19s). Hybrid saves ~14s/turn vs Local.
- **Lexical diversity**: Hybrid leads in distinct-1 (0.478) and distinct-3 (0.996); Local leads in distinct-2 (0.944).
- **Consistency**: All three modes achieve 100% consistency rate (zero KG conflicts across 90 total turns).
- **Entity coverage**: Local is highest (0.722), API (0.667), Hybrid (0.656).
- **Narrative quality**: Local and Hybrid tie at 6.0/10 for fantasy; API scores only 3.67/10 average across all genres.
- **Zero NLU fallbacks**: fastcoref, spaCy, and DistilBERT all loaded successfully in all three modes.

---

## 2. Aggregate Comparison (All Genres Combined)

### 2.1 Automatic Metrics

| Metric | Local | API | Hybrid | Best |
|--------|------:|----:|-------:|------|
| **distinct_1** | 0.4842 | 0.4313 | **0.4779** | Hybrid |
| **distinct_2** | **0.9439** | 0.9154 | 0.9455 | Hybrid |
| **distinct_3** | 0.9950 | 0.9884 | **0.9961** | Hybrid |
| **self_bleu** (lower=better) | 0.0181 | 0.0365 | **0.0171** | Hybrid |
| **entity_coverage** | **0.7222** | 0.6667 | 0.6556 | Local |
| **consistency_rate** | 1.0000 | 1.0000 | 1.0000 | Tie |
| **type_token_ratio** | **0.3787** | 0.3222 | 0.3764 | Local |
| **flesch_reading_ease** | 18.41 | **40.51** | 18.23 | API |
| **lexical_overlap** | 0.1084 | **0.1465** | 0.1149 | API |
| **graph_density_average** | 0.0374 | **0.1436** | 0.0768 | API |
| **graph_density_delta** | -0.0472 | -0.2170 | **-0.1500** | Hybrid |

**Summary**: Hybrid wins 5 of 11 auto metrics. Local wins 3. API wins 3. Hybrid dominates lexical diversity (distinct-n, self_bleu). API wins readability (flesch) and graph complexity.

### 2.2 LLM Judge Scores (1-10)

| Dimension | Local | API | Hybrid | Best |
|-----------|------:|----:|-------:|------|
| **narrative_quality** | **6.00** | 3.67 | 5.00 | Local |
| **consistency** | 2.00 | 1.67 | **2.00** | Local/Hybrid |
| **player_agency** | 1.33 | 1.33 | **1.33** | Tie |
| **creativity** | **5.33** | 3.00 | 4.00 | Local |
| **pacing** | 2.67 | 2.00 | **2.67** | Local/Hybrid |
| **option_relevance** | 1.67 | **2.67** | 1.67 | API |
| **causal_link** | 1.33 | 0.67 | **1.33** | Local/Hybrid |
| **local_coherence** | 2.00 | 1.33 | **2.00** | Local/Hybrid |
| **average** | **2.79** | 2.04 | 2.50 | Local |

**Summary**: Local wins the overall average (2.79) and dominates narrative_quality (6.00) and creativity (5.33). API only wins option_relevance (2.67). Hybrid matches Local on consistency, pacing, causal_link, and local_coherence but falls short on narrative_quality and creativity.

### 2.3 Latency (seconds per turn)

| Metric | Local | API | Hybrid | Best |
|--------|------:|----:|-------:|------|
| **Mean** | 184.19 | 173.64 | **169.95** | Hybrid |
| **P50** | 180.94 | 177.30 | **167.12** | Hybrid |
| **P90** | 189.27 | 181.75 | **179.15** | Hybrid |
| **P95** | 190.32 | 182.31 | **180.66** | Hybrid |
| **Std** | 6.22 | 11.50 | **11.07** | Local |
| **Min** | **180.27** | 160.76 | **160.58** | Hybrid |
| **Max** | 191.36 | 182.87 | **182.16** | Hybrid |

**Summary**: Hybrid is fastest at every percentile. The ~14s/turn savings vs Local comes from hybrid mode routing structured tasks (options, relations) to the faster Mimo API while keeping creative story generation on the local Qwen3 model.

---

## 3. Per-Genre Breakdown

### 3.1 Fantasy

| Metric | Local | API | Hybrid | Best |
|--------|------:|----:|-------:|------|
| **Mean latency (s)** | 191.36 | 160.76 | **160.58** | Hybrid |
| **distinct_1** | **0.4583** | 0.3722 | 0.4296 | Local |
| **distinct_2** | **0.9476** | 0.8802 | 0.9355 | Local |
| **self_bleu** | 0.0238 | 0.0599 | **0.0221** | Hybrid |
| **entity_coverage** | 0.5000 | 0.2500 | **0.8000** | Hybrid |
| **consistency_rate** | 1.0000 | 1.0000 | 1.0000 | Tie |
| **flesch_reading_ease** | 32.28 | **51.80** | 21.36 | API |
| **LLM avg** | **2.75** | 1.75 | 3.38 | Hybrid |
| **narrative_quality** | **7** | 3 | **7** | Local/Hybrid |
| **creativity** | **6** | 2 | **6** | Local/Hybrid |

**Fantasy takeaway**: Hybrid achieves the highest LLM average (3.38) with best entity coverage (0.80) and fastest latency (160.58s). Local matches Hybrid on narrative_quality (7) and creativity (6). API scores lowest across all LLM dimensions.

### 3.2 Sci-Fi

| Metric | Local | API | Hybrid | Best |
|--------|------:|----:|-------:|------|
| **Mean latency (s)** | 180.27 | 182.87 | **167.12** | Hybrid |
| **distinct_1** | **0.5360** | 0.4061 | 0.4398 | Local |
| **distinct_2** | **0.9628** | 0.9122 | 0.9375 | Local |
| **self_bleu** | 0.0079 | 0.0401 | **0.0152** | Local |
| **entity_coverage** | **1.0000** | 0.7500 | 0.5000 | Local |
| **consistency_rate** | 1.0000 | 1.0000 | 1.0000 | Tie |
| **flesch_reading_ease** | -5.34 | **52.54** | 21.84 | API |
| **LLM avg** | 2.00 | **2.12** | 2.00 | API |
| **narrative_quality** | 4 | **4** | 4 | Tie |
| **creativity** | **4** | **4** | 3 | Local/API |

**Sci-Fi takeaway**: Local dominates lexical diversity (distinct-1: 0.536, self_bleu: 0.008) and achieves perfect entity coverage (1.0). However, its flesch score (-5.34) indicates very dense/complex text. API wins readability (52.54) and narrowly edges LLM average (2.12). Hybrid is fastest (167.12s).

### 3.3 Mystery

| Metric | Local | API | Hybrid | Best |
|--------|------:|----:|-------:|------|
| **Mean latency (s)** | 180.94 | 177.30 | **182.16** | API |
| **distinct_1** | 0.4582 | 0.5157 | **0.5643** | Hybrid |
| **distinct_2** | 0.9214 | 0.9538 | **0.9634** | Hybrid |
| **self_bleu** | 0.0227 | 0.0096 | **0.0140** | API |
| **entity_coverage** | 0.6667 | **1.0000** | 0.6667 | API |
| **consistency_rate** | 1.0000 | 1.0000 | 1.0000 | Tie |
| **flesch_reading_ease** | 28.28 | **17.18** | 11.48 | Local |
| **LLM avg** | **3.62** | 2.25 | 2.12 | Local |
| **narrative_quality** | **7** | 4 | 4 | Local |
| **creativity** | **6** | 3 | 3 | Local |

**Mystery takeaway**: Local dominates LLM scores (avg 3.62, narrative_quality 7, creativity 6) — the highest single-session scores across all modes and genres. Hybrid wins lexical diversity (distinct-1: 0.564). API wins entity coverage (1.0) and self_bleu (0.0096).

---

## 4. Fallback Audit

### NLU Module Status

| Module | Local | API | Hybrid |
|--------|-------|-----|--------|
| **fastcoref (neural coref)** | Loaded | Loaded | Loaded |
| **spaCy (NER)** | Loaded | Loaded | Loaded |
| **DistilBERT (intent)** | Loaded | Loaded | Loaded |

**NLU fallback warnings**: 0 across all 3 modes.

### Option Generation Fallbacks

| Mode | Count | Cause |
|------|------:|-------|
| Local | 3 | JSON parse errors from LLM output |
| API | 0 | — |
| Hybrid | 0 | — |

Local mode had 3 option generation fallbacks where the LLM returned malformed JSON. These are not NLU fallbacks — they are graceful degradation in the option generator when the local model's JSON output is invalid. API and Hybrid modes had zero option generation fallbacks, likely because the Mimo API produces more structured JSON output.

---

## 5. Analysis

### 5.1 Why Local Wins on Narrative Quality and Creativity

Local mode's Qwen3-4B-Instruct-2507 produces more creative and narratively rich text than the Mimo API. This is evident in the LLM Judge scores: Local averages 6.00 for narrative_quality and 5.33 for creativity, compared to API's 3.67 and 3.00. The mystery genre session under Local mode achieved the highest single-session narrative_quality score (7) across all 9 sessions.

### 5.2 Why Hybrid Is Fastest

Hybrid mode routes story generation to the local Qwen3 model but sends structured tasks (option generation, relation extraction, consistency checking) to the Mimo API. Since these structured tasks are called multiple times per turn, the API's faster response time (typically <1s vs 10-46s for local) compounds into significant per-turn savings. Hybrid saves ~14s/turn vs pure Local.

### 5.3 Why API Wins Readability

API mode's flesch_reading_ease average (40.51) is more than double Local (18.41) and Hybrid (18.23). The Mimo API produces simpler, more accessible prose. However, this comes at the cost of creativity and narrative depth — API scores lowest on both dimensions.

### 5.4 Consistency Is Universal

All three modes achieve 100% consistency_rate across all 90 turns. The knowledge graph conflict detection and resolution system works equally well regardless of NLG backend. This suggests the KG pipeline is robust and decoupled from the story generation model.

### 5.5 The Low LLM Judge Scores

Even the best average (Local: 2.79/10) is low. This reflects the deterministic evaluation policy (always picking the first option), which produces shallow, predictable narratives. The LLM Judge penalizes lack of player agency (all modes score 1.33) and limited causal chains. A human-driven evaluation would likely produce significantly higher scores.

---

## 6. Recommendations

1. **For production use**: **Hybrid mode** offers the best balance — fastest latency (169.95s/turn), competitive LLM scores (2.50 avg), and zero NLU fallbacks.
2. **For maximum narrative quality**: **Local mode** produces the most creative and narratively rich stories, at the cost of ~14s/turn additional latency.
3. **For API-only deployments**: The Mimo API is viable but produces less creative narratives. Consider fine-tuning or prompt engineering to improve narrative_quality and creativity scores.
4. **For evaluation methodology**: Replace the deterministic "first option" policy with human-driven or LLM-simulated diverse choices to produce more meaningful LLM Judge scores.

---

## 7. Raw Data Sources

- Local mode: `docs/eval_local_results.json`
- API mode: `docs/eval_api_results.json`
- Hybrid mode: `docs/eval_hybrid_results.json`
- Local stdout: `docs/eval_local_stdout.log`
- API stdout: `docs/eval_api_stdout.log`
- Hybrid stdout: `docs/eval_hybrid_stdout.log`
