# Hybrid Model Strategy - Detailed Implementation Guide

**Based on:** Model Comparison v2 Report  
**Date:** March 31, 2026

---

## 🎯 Why Hybrid?

Our evaluation revealed that each model has distinct strengths:

| Task | Local Qwen3-4B | Xiaomi Mimo v2 Flash |
|------|----------------|----------------------|
| Creative narrative | **9.12/10** (LLM Judge) | 8.38/10 |
| JSON generation | ❌ FAIL | ✅ PASS |
| Speed | 9.05s avg | **3.78s avg** |

**Key insight:** The local model produces higher quality creative writing but fails at structured JSON output. The Mimo model is faster and reliable for JSON but scores lower on creative quality.

---

## 🏗️ Architecture: Task-Based Model Routing

### Current Architecture (Single Model)
```
┌─────────────────────────────────────────────────────────┐
│                    Game Engine                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Story Gen    │  │ Option Gen   │  │ KG Extract   │  │
│  │ (LLM call)   │  │ (LLM call)   │  │ (LLM call)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                ┌───────▼───────┐
                │  Single LLM   │
                │  (one model)  │
                └───────────────┘
```

### Proposed Hybrid Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Game Engine                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Story Gen    │  │ Option Gen   │  │ KG Extract   │  │
│  │ (Creative)   │  │ (Structured) │  │ (Structured) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
        │                    │                │
        ▼                    ▼                ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Local Qwen3-4B│    │ Xiaomi Mimo   │    │ Xiaomi Mimo   │
│ (Port 8000)   │    │ (Cloud API)   │    │ (Cloud API)   │
│ 9.12 quality  │    │ JSON reliable │    │ JSON reliable │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## 📋 Task Classification Matrix

| Task | Model | Rationale |
|------|-------|-----------|
| **Opening scene** | Local Qwen3-4B | First impression matters; creative quality critical |
| **Story continuation** | Local Qwen3-4B | Narrative flow; LLM Judge 9.12/10 |
| **Option generation** | Xiaomi Mimo | Requires strict JSON; 8x faster |
| **KG relation extraction** | Xiaomi Mimo | Requires strict JSON; reliability critical |
| **Conflict checking** | Xiaomi Mimo | Fast structured responses needed |

---

## 🔧 Implementation Options

### Option 1: Dual Client Configuration (Recommended)

Create separate clients for different task types:

```python
# src/utils/model_router.py

from src.utils.api_client import LLMClient

# Creative tasks client (local model)
creative_client = LLMClient(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="merged_model_Qwen3-4B-Instruct-2507"
)

# Structured tasks client (Mimo API)
structured_client = LLMClient(
    api_key="sk-sqnijikyy32vhh0ga5u1qae3wyhfenlezunewtexdcub0s1u",
    base_url="https://api.xiaomimimo.com/v1",
    model="mimo-v2-flash"
)

def get_client(task_type: str) -> LLMClient:
    if task_type in ["story", "opening", "continuation"]:
        return creative_client
    else:
        return structured_client
```

### Option 2: Environment-Based Switching

Add configuration for model routing:

```python
# config.py additions

class Settings(BaseSettings):
    # Existing settings...
    
    # Model routing
    CREATIVE_MODEL_URL: str = "http://localhost:8000/v1"
    CREATIVE_MODEL: str = "merged_model_Qwen3-4B-Instruct-2507"
    
    STRUCTURED_MODEL_URL: str = "https://api.xiaomimimo.com/v1"
    STRUCTURED_MODEL: str = "mimo-v2-flash"
    STRUCTURED_MODEL_KEY: str = "sk-..."
```

### Option 3: Automatic Fallback

Use local model first, fall back to Mimo on JSON failures:

```python
def generate_options_with_fallback(story_text, kg_summary):
    try:
        # Try local model first (better quality)
        return local_client.chat_json(messages)
    except JSONParseError:
        # Fall back to Mimo (reliable JSON)
        logger.warning("Local JSON failed, using Mimo fallback")
        return mimo_client.chat_json(messages)
```

---

## 📊 Expected Performance (Hybrid)

| Metric | Current (Single) | Hybrid | Improvement |
|--------|------------------|--------|-------------|
| Story quality | 8.38 (Mimo) | **9.12 (Local)** | +8.8% |
| JSON reliability | 100% (Mimo) | 100% (Mimo) | Same |
| Avg latency | 3.78s (Mimo) | ~6s (mixed) | -58% slower |
| Cost | API fees only | Mixed | 50% reduction* |

*Assuming 50% of calls are creative (local) and 50% structured (API)

---

## 🚀 Implementation Roadmap

### Phase 1: Configuration (30 min)
1. Add dual model settings to `config.py`
2. Update `.env` with both model configurations
3. Create model router utility

### Phase 2: Integration (1 hour)
1. Modify `story_generator.py` to use creative client
2. Modify `option_generator.py` to use structured client
3. Modify `relation_extractor.py` to use structured client

### Phase 3: Testing (30 min)
1. Run evaluation with hybrid configuration
2. Verify JSON reliability
3. Measure combined quality scores

---

## 💡 Specific Code Changes

### 1. Update `story_generator.py`

```python
class StoryGenerator:
    def generate_opening(self, genre: str = "fantasy") -> str:
        from src.utils.model_router import get_client
        
        # Use creative model for narrative
        client = get_client("story")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OPENING_PROMPT.format(genre=genre)},
        ]
        return client.chat(messages)
```

### 2. Update `option_generator.py`

```python
class OptionGenerator:
    def generate(self, story_text, kg_summary, num_options=None):
        from src.utils.model_router import get_client
        
        # Use structured model for JSON
        client = get_client("json")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OPTION_GENERATION_PROMPT.format(...)},
        ]
        data = client.chat_json(messages)
        # ... rest of processing
```

---

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Local server downtime | Story generation fails | Add Mimo fallback for creative tasks |
| API key expiry | JSON tasks fail | Keep local model as backup |
| Increased complexity | Harder to debug | Add logging for model selection |
| Latency inconsistency | Variable UX | Cache common responses |

---

## 📈 Success Metrics

After implementing hybrid approach, measure:

1. **Story quality**: LLM Judge score should remain ≥9.0
2. **JSON reliability**: Should maintain 100% parse success
3. **User experience**: Turn latency should stay <10s
4. **Cost reduction**: API calls should decrease by ~50%

---

## 🎯 Final Recommendation

**Implement Option 1 (Dual Client Configuration)** because:

1. **Clear separation**: Each client has a single responsibility
2. **Easy to test**: Can test each model independently
3. **Simple rollback**: Can switch back to single model easily
4. **Incremental**: Can implement one component at a time

**Start with:**
- Story generation → Local model (immediate quality boost)
- Option generation → Mimo (immediate reliability boost)
- KG extraction → Mimo (already working)

This hybrid approach gives you the **best of both worlds**: superior creative quality from the local model and reliable structured output from the cloud API.
