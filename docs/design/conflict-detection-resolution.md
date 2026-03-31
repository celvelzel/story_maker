# Conflict Detection & Resolution

> **Last Updated:** 2026-04-01  
> **Module:** `src/knowledge_graph/conflict_detector.py`

## 1. Overview

The Conflict Detection & Resolution system ensures narrative consistency in StoryWeaver's Knowledge Graph. It operates as a multi-layer detection pipeline with configurable resolution strategies.

## 2. Detection Architecture

Three detection layers operate sequentially:

```
┌─────────────────────────────────────────────────────┐
│              ConflictDetector.check_all()            │
│                                                     │
│  Layer 1: Rule-Based (Deterministic)                │
│  ├── Exclusive relation pairs (ally_of ↔ enemy_of)  │
│  └── Dead-active detection                          │
│                                                     │
│  Layer 1b: Temporal (Deterministic)                 │
│  ├── Post-death entity actions                      │
│  └── Causal inversion (causes/enables)              │
│                                                     │
│  Layer 2: LLM-Based (Probabilistic)                 │
│  └── Logical contradiction analysis                 │
│      └── Confidence partitioning:                   │
│          ≥ 0.75 → Accept                            │
│          0.45-0.74 → Defer                          │
│          < 0.45 → Drop                              │
└─────────────────────────────────────────────────────┘
```

### 2.1 Rule-Based Detection

Detects deterministic contradictions in the graph:

| Conflict Type | Detection Logic |
|---------------|----------------|
| `exclusive_relation` | Same source→target has both `ally_of` and `enemy_of`, or `alive` and `dead` |
| `dead_active` | Entity marked `dead` still has active relations (`possesses`, `located_at`, `ally_of`) |

### 2.2 Temporal Detection

Detects time-based contradictions:

| Conflict Type | Detection Logic |
|---------------|----------------|
| `dead_entity_action` | Entity created relations after its death turn |
| `causal_inversion` | Effect entity created before cause entity for `causes`/`enables` relations |

### 2.3 LLM-Based Detection

Sends the current world state and new story text to an LLM for logical contradiction analysis. Returns JSON with conflict descriptions and confidence scores.

**Confidence Partitioning:**
- **≥ 0.75**: Accepted for resolution
- **0.45–0.74**: Deferred (tracked but not resolved)
- **< 0.45**: Dropped (too noisy)

## 3. Resolution Strategies

Two strategies are available via `KG_CONFLICT_RESOLUTION` setting:

### 3.1 `keep_latest` (Deterministic)

| Conflict Type | Resolution |
|---------------|------------|
| `exclusive_relation` | Remove the relation with older `last_confirmed_turn` |
| `dead_active` | Remove the active relation from dead entity |
| `temporal` | Left unresolved (requires manual intervention) |
| `llm` | Left unresolved (cannot deterministically resolve) |

### 3.2 `llm_arbitrate` (Hybrid)

**Deterministic-first pass:** Resolves `exclusive_relation` and `dead_active` conflicts using `KeepLatestResolver` logic.

**LLM arbitration:** For LLM-detected conflicts with confidence ≥ 0.75, calls LLM to decide resolution:
- `keep_new` — Remove older relation
- `keep_old` — Remove newer relation
- `remove_relation` — Remove specific relation
- `update_entity` — Update entity status
- `no_action` — Not a real conflict

**Temporal conflicts:** Always left unresolved for explicit follow-up.

## 4. Conflict Output

Each detected conflict is a dictionary:

```python
{
    "type": "exclusive_relation" | "dead_active" | "temporal" | "llm",
    "source": str,        # Source entity
    "target": str,        # Target entity
    "description": str,   # Human-readable description
    "confidence": str,    # Confidence score (for LLM conflicts)
    # Additional fields per type:
    "relation_a": str,    # For exclusive_relation
    "relation_b": str,    # For exclusive_relation
    "subtype": str,       # For temporal: "dead_entity_action" | "causal_inversion"
    "death_turn": int,    # For temporal
    "relation_turn": int, # For temporal
}
```

## 5. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KG_CONFLICT_RESOLUTION` | `llm_arbitrate` | `keep_latest` or `llm_arbitrate` |

### Internal Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `LLM_CONFLICT_ACCEPT_THRESHOLD` | `0.75` | Minimum confidence to accept LLM conflict |
| `LLM_CONFLICT_DEFER_LOW` | `0.45` | Lower bound for deferral band |
| `LLM_CONFLICT_DEFER_HIGH` | `0.74` | Upper bound for deferral band |

## 6. Usage

```python
from src.knowledge_graph.conflict_detector import ConflictDetector, get_resolver

# Create detector
detector = ConflictDetector(kg)

# Run all detection layers
conflicts = detector.check_all(new_story_text)

# Resolve with configured strategy
resolver = get_resolver("llm_arbitrate")
unresolved = resolver.resolve(conflicts, kg)
```

---
*Related: [entity-importance.md](entity-importance.md) for KG importance scoring.*
