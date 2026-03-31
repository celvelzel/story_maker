# Xiaomi Mimo v2 Flash Model Evaluation Report

**Model:** mimo-v2-flash
**Base URL:** https://api.xiaomimimo.com/v1
**Provider:** Xiaomi Mimo API (Cloud)

---

## ⏱️ Performance

- **Story generation latency:** 29.24s
- **JSON generation latency:** 1.11s
- **Average turn latency:** 3.78s

## 📊 Quality Metrics

- **Distinct-1:** 0.7391
- **Distinct-2:** 0.9825
- **Type-Token Ratio:** 0.7328
- **Flesch Reading Ease:** 75.63

## ✅ Functionality

- **JSON parsing:** PASS - Correctly generates single JSON objects

## ⚖️ LLM Judge Scores (from full evaluation)

- **narrative_quality:** 9
- **consistency:** 9
- **player_agency:** 7
- **creativity:** 9
- **pacing:** 8
- **option_relevance:** 8
- **causal_link:** 8
- **local_coherence:** 9
- **average:** 8.38

## 📖 Sample Generated Story

```
You wake to the chill of the riverbank, mist coiling across the water and the sky still bruised with night. A silver feather drifts onto your chest, humming faintly like a distant bell. When your fingers close around it, the bell becomes a voice: Choose, and the world will answer.

A path of pale stones raises itself from the mud, leading toward a copse of trees that bend like listening ears. On the other side, a fox with ember-tipped fur trots from the reeds, tail writing a question mark in the mist. The voice says nothing more, but the feather's hum grows warmer, as if urging you forward.
```

## 🔍 Observations

1. **Latency:** Story generation ~29s (cloud API), JSON generation very fast ~1s
2. **Quality:** Excellent distinct-2 score (0.98), good vocabulary diversity
3. **JSON:** Perfect JSON formatting - passes all parsing tests
4. **Consistency:** Maintains coherent multi-turn conversations
5. **LLM Judge:** High scores across all dimensions (avg 8.38/10)
