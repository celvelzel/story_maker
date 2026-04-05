# Model Comparison Report v2 (with LLM Judge Comparison)

> **Last Updated**: 2026-03-31

**Evaluation Date:** March 31, 2026
**Task:** Interactive Fiction Story Generation
**Models Compared:** Local Qwen3-4B-Instruct vs Xiaomi Mimo v2 Flash

---

## 📊 Executive Summary

| Aspect | Local Qwen3-4B | Xiaomi Mimo v2 Flash | Winner |
|--------|----------------|----------------------|--------|
| **Deployment** | Local CUDA GPU | Cloud API | Depends on needs |
| **Avg Turn Latency** | 9.05s | 3.78s | 🏆 Mimo |
| **Quality (distinct-2)** | 0.9956 | 0.9825 | 🏆 Local |
| **JSON Parsing** | ❌ FAIL | ✅ PASS | 🏆 Mimo |
| **LLM Judge Average** | 9.12/10 | 8.38/10 | 🏆 Local |
| **Cost** | Free (GPU electricity) | API usage fees | 🏆 Local |
| **Privacy** | Full data privacy | Data sent to cloud | 🏆 Local |

---

## ⏱️ Performance Comparison

| Metric | Local Qwen3-4B | Xiaomi Mimo v2 Flash | Notes |
|--------|----------------|----------------------|-------|
| Story generation | 53.11s (first) / ~9s (subsequent) | 29.24s | Local has warmup cost |
| JSON generation | 9.06s | 1.11s | Mimo 8x faster |
| Average turn latency | 9.05s | 3.78s | Mimo 2.4x faster |

**Analysis:** The Mimo API is significantly faster for JSON generation and slightly faster for story generation. However, the local model's latency is acceptable for interactive gameplay (~9s per turn).

---

## 📊 Quality Metrics Comparison

| Metric | Local Qwen3-4B | Xiaomi Mimo v2 Flash | Interpretation |
|--------|----------------|----------------------|----------------|
| **Distinct-1** | 0.7555 | 0.7391 | Local slightly better |
| **Distinct-2** | 0.9956 | 0.9825 | Local slightly better |
| **Type-Token Ratio** | 0.7186 | 0.7328 | Mimo slightly better |
| **Flesch Reading Ease** | 72.57 | 75.63 | Both standard/easy |

**Analysis:** Both models show excellent text diversity (distinct-2 > 0.98). The local model has marginally higher distinct-2 scores, suggesting slightly more varied vocabulary use.

---

## ⚖️ LLM Judge Scores Comparison

| Dimension | Local Qwen3-4B | Xiaomi Mimo v2 Flash | Winner |
|-----------|----------------|----------------------|--------|
| **narrative_quality** | 10 | 9 | 🏆 Local |
| **consistency** | 10 | 9 | 🏆 Local |
| **player_agency** | 8 | 7 | 🏆 Local |
| **creativity** | 9 | 9 | Tie |
| **pacing** | 9 | 8 | 🏆 Local |
| **option_relevance** | 7 | 8 | 🏆 Mimo |
| **causal_link** | 10 | 8 | 🏆 Local |
| **local_coherence** | 10 | 9 | 🏆 Local |
| **AVERAGE** | **9.12** | **8.38** | 🏆 Local |

**Surprising Finding:** The local model achieved higher LLM judge scores than the Mimo model, particularly excelling in narrative quality, consistency, and causal link. The Mimo model scored slightly better only in option_relevance.

---

## ✅ Functionality Comparison

| Feature | Local Qwen3-4B | Xiaomi Mimo v2 Flash |
|---------|----------------|----------------------|
| Story generation | ✅ Works | ✅ Works |
| Multi-turn conversation | ✅ Works | ✅ Works |
| JSON option generation | ❌ FAIL (multiple objects) | ✅ PASS |
| Instruction following | ⚠️ Sometimes verbose | ✅ Good |

**Analysis:** The Mimo model significantly outperforms the local model in JSON generation and instruction following. The local model struggles with strict formatting requirements.

---

## 💰 Cost Analysis

| Factor | Local Qwen3-4B | Xiaomi Mimo v2 Flash |
|--------|----------------|----------------------|
| **Initial setup** | GPU hardware required | API key only |
| **Per-request cost** | Electricity (~$0.001) | API fees (varies) |
| **Scaling** | Limited by GPU memory | Unlimited |
| **Offline capability** | ✅ Yes | ❌ No |

---

## 🔒 Privacy & Control

| Aspect | Local Qwen3-4B | Xiaomi Mimo v2 Flash |
|--------|----------------|----------------------|
| Data privacy | ✅ Full control | ❌ Data sent to cloud |
| Customization | ✅ Can fine-tune | ❌ Limited |
| Availability | ⚠️ Depends on GPU | ✅ Always available |

---

## 📈 Final Verdict

### Overall Winner by Category:

1. **Creative Quality:** 🏆 **Local Qwen3-4B** (LLM Judge: 9.12 vs 8.38)
2. **Speed:** 🏆 **Xiaomi Mimo v2 Flash** (3.78s vs 9.05s)
3. **Reliability:** 🏆 **Xiaomi Mimo v2 Flash** (JSON works)
4. **Cost:** 🏆 **Local Qwen3-4B** (free vs API fees)
5. **Privacy:** 🏆 **Local Qwen3-4B** (local vs cloud)

---

## 🎯 Recommendations

### Choose Local Qwen3-4B if:
- **Creative quality is top priority** (higher LLM judge scores)
- Data privacy is critical
- You need offline capability
- You want to fine-tune the model
- Cost per request matters at scale

### Choose Xiaomi Mimo v2 Flash if:
- You need reliable JSON generation
- Latency is critical
- You prefer managed infrastructure
- Instruction following is important

### Hybrid Approach (Recommended):
- Use **Local model for creative story generation** (higher quality)
- Use **Mimo for structured tasks** (JSON generation, option creation)
- This combines the strengths of both models

---

## 📁 Report Files

1. `reports/local_model_eval_report.md` - Original local model evaluation
2. `reports/local_model_eval_v2_report.md` - Local model with LLM judge (NEW)
3. `reports/mimo_eval_report.md` - Mimo model evaluation
4. `reports/model_comparison_report.md` - Original comparison
5. `reports/model_comparison_v2_report.md` - This report with LLM judge comparison (NEW)
