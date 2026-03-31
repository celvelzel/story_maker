# Automated Test Report

**Generated At**: 2026-03-26 16:09:19  
**Goal**: Evaluate automated metrics, LLM-based subjective scoring, and end-to-end response times across different story genres.

## 1. Test Configuration
- **Total Sessions**: 3
- **Genres Tested**: fantasy, sci-fi, mystery
- **Max Turns per Session**: 10
- **Warmup Turns**: 0
- **Action Strategy**: Automatically select the first option for every turn.

## 2. Session Details

| Session | Genre | Turns | Mean (s) | P50 (s) | P90 (s) | P95 (s) | Std (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | fantasy | 10 | 18.9360 | 18.7056 | 22.2735 | 23.3086 | 3.0632 | OK |
| 2 | sci-fi | 10 | 19.6930 | 19.2864 | 23.3718 | 24.1221 | 3.0941 | OK |
| 3 | mystery | 10 | 20.6482 | 20.3414 | 22.7168 | 22.8156 | 1.4706 | OK |

## 3. Aggregate Results

### 3.1 Response Time Performance

| Metric | Value (s) |
|---|---:|
| **Mean** | 19.7590 |
| **P50** | 19.6930 |
| **P90** | 20.4571 |
| **P95** | 20.5527 |
| **Std Dev** | 0.8580 |
| **Min** | 18.9360 |
| **Max** | 20.6482 |

### 3.2 Automated Metrics (`metrics.py`)

| Metric | Value | Description |
|---|---:|---|
| **Distinct-1** | 0.3493 | Unigram diversity (higher is better) |
| **Distinct-2** | 0.7083 | Bigram diversity |
| **Distinct-3** | 0.8921 | Trigram diversity |
| **Self-BLEU** | 0.2398 | Redundancy measure (lower is better) |
| **Entity Coverage** | 0.9630 | Ratio of KG entities mentioned in text |
| **Consistency Rate**| 1.0000 | Percentage of turns without KG conflicts |
| **Type-Token Ratio**| 0.3060 | Vocabulary richness |
| **Flesch Reading Ease** | 53.5867 | Readability score |
| **Lexical Overlap** | 0.2817 | Word overlap between adjacent turns |
| **Avg Graph Density**| 0.1084 | Average world complexity |
| **Graph Density Delta**| 0.0241 | Complexity growth over session |

### 3.3 LLM Judge Results (`llm_judge.py`)

| Dimension | Score (1-10) |
|---|---:|
| **Narrative Quality** | 8.00 |
| **Consistency** | 6.67 |
| **Player Agency** | 6.67 |
| **Creativity** | 6.33 |
| **Pacing** | 8.00 |
| **Option Relevance** | 7.33 |
| **Causal Link** | 6.33 |
| **Local Coherence** | 6.00 |
| **Average** | **6.91** |

## 4. Metric Interpretations
- **Response Time**: Measures the duration from player input to full UI update (including NLU, NLG, and KG steps).
- **Consistency Rate**: High values indicate that the conflict detection and resolution strategies are maintaining a stable world state.
- **Entity Coverage**: High coverage ensures the story reflects the underlying knowledge graph accurately.
- **LLM Judge**: Provides a multi-dimensional subjective quality assessment. `Narrative Quality` and `Pacing` are currently the highest-rated areas.

## 5. Notes
- **Errors**: 0 sessions failed.
- **Warmup**: The lack of warmup turns means cold-start latency for models (like DistilBERT) is included in the "Mean" response time.
