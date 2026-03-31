# Model Comparison Report: Local Qwen3-4B vs Xiaomi Mimo v2 Flash

> **Last Updated**: 2026-03-31

**Evaluation Date:** March 31, 2026
**Task:** Interactive Fiction Story Generation

---

## 📊 Executive Summary

| Aspect | Local Qwen3-4B | Xiaomi Mimo v2 Flash | Winner |
|--------|----------------|----------------------|--------|
| **Deployment** | Local CUDA GPU | Cloud API | Depends on needs |
| **Latency (avg)** | 9.05s | 3.78s | 🏆 Mimo |
| **Quality (distinct-2)** | 0.9956 | 0.9825 | 🏆 Local |
| **JSON Parsing** | FAIL | PASS | 🏆 Mimo |
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

## 📝 LLM Judge Scores (from full evaluation)

| Dimension | Local Qwen3-4B | Xiaomi Mimo v2 Flash |
|-----------|----------------|----------------------|
| Narrative quality | N/A | 9/10 |
| Consistency | N/A | 9/10 |
| Player agency | N/A | 7/10 |
| Creativity | N/A | 9/10 |
| Pacing | N/A | 8/10 |
| Option relevance | N/A | 8/10 |
| Causal link | N/A | 8/10 |
| Local coherence | N/A | 9/10 |
| **Average** | **N/A** | **8.38/10** |

*Note: LLM judge scores were only collected for Mimo model during full evaluation.*

---

## 🎯 Recommendations

### Choose Local Qwen3-4B if:
- Data privacy is critical
- You need offline capability
- You want to fine-tune the model
- Cost per request matters at scale

### Choose Xiaomi Mimo v2 Flash if:
- You need reliable JSON generation
- Latency is critical
- You prefer managed infrastructure
- Instruction following is important

### Hybrid Approach:
- Use **Mimo for structured tasks** (JSON generation, option creation)
- Use **Local model for creative tasks** (story generation where privacy matters)
- This combines the strengths of both models

---

## 📁 Report Files

1. `reports/local_model_eval_report.md` - Detailed local model evaluation
2. `reports/mimo_eval_report.md` - Detailed Mimo model evaluation
3. `reports/model_comparison_report.md` - This comparison report
