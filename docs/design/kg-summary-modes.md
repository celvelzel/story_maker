# KG Summary Modes

> **Last Updated:** 2026-04-01  
> **Module:** `src/knowledge_graph/graph.py`

## 1. Overview

The Knowledge Graph summary is a textual representation of the current world state, consumed by the LLM during story and option generation. StoryWeaver supports two summary modes: `flat` (backward compatible) and `layered` (importance-ranked).

## 2. Mode Comparison

| Feature | Flat Mode | Layered Mode |
|---------|-----------|--------------|
| **Organization** | Sequential list | Importance-tiered sections |
| **Entity Info** | Basic (name, type, attributes) | Rich (description, status, history, relations) |
| **Sorting** | Insertion order | Importance score (descending) |
| **Timeline** | Not included | Included (recent events) |
| **Token Efficiency** | Lower (all entities equal) | Higher (prioritizes important entities) |
| **Best For** | Small KGs, debugging | Production, large KGs |

## 3. Flat Mode

Generates a simple sequential listing of entities and relations.

**Output Format:**
```
=== World State ===
- Dragon [creature] {'created_turn': 0, 'last_mentioned_turn': 3, ...}
- Forest [location] {'created_turn': 0, ...}
- Sword [item] {'created_turn': 1, ...}

=== Relations ===
- dragon --[located_at]--> forest
- dragon --[enemy_of]--> player
- player --[possesses]--> sword
```

**Use Cases:**
- Debugging and development
- Small KGs where entity count is manageable
- Backward compatibility with existing prompts

## 4. Layered Mode

Generates an importance-ranked summary with rich entity details and a timeline.

**Output Format:**
```
=== Core Entities (High Importance) ===
- Dragon [creature] (importance: 0.85, turn 3)
  Description: A fearsome red dragon guarding the treasure
  Status: {health: injured, mood: aggressive}
  Emotion: fearful
  History: turn 2: health=healthy→injured; turn 3: mood=calm→aggressive
  Relations: located_at→forest (0.9), enemy_of→player (0.8)

=== Secondary Entities ===
- Forest [location] (importance: 0.52, turn 3)
  Description: A dark, ancient forest
  Relations: contains→dragon (0.7)

=== Background ===
- Tavern [location] (importance: 0.15, last seen turn 0)
- Innkeeper [person] (importance: 0.12, last seen turn 0)

=== Recent Timeline ===
- Turn 3: Player attacked the dragon with their sword
- Turn 2: Dragon breathed fire at the player
- Turn 1: Player entered the dark forest
```

**Tier Thresholds:**
- **Core:** importance ≥ 0.6
- **Secondary:** 0.3 ≤ importance < 0.6
- **Background:** importance < 0.3

**Entity Block Components:**
- Name and type with importance score
- Description (if available)
- Current status (if available)
- Last associated emotion (if available)
- Recent status history (last 3 entries)
- Outgoing relations with confidence scores

**Timeline:**
- Shows the most recent `KG_MAX_TIMELINE_ENTRIES` (default: 5) relation confirmations
- Sorted by `last_confirmed_turn` descending

## 5. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KG_SUMMARY_MODE` | `layered` | `flat` or `layered` |
| `KG_MAX_TIMELINE_ENTRIES` | `5` | Maximum entries in layered timeline |

## 6. Switching Modes

### Via Configuration
```env
KG_SUMMARY_MODE=layered
```

### Via Code
```python
from src.engine.game_engine import GameEngine
engine = GameEngine(summary_mode="layered")
```

### Via Frontend
In the Streamlit sidebar **"⚙ KG Strategy Settings"** panel, select the **"Summary Mode"** dropdown.

## 7. Performance Considerations

- **Flat mode** is faster to generate (simple iteration) but may waste tokens on irrelevant entities.
- **Layered mode** requires sorting by importance and formatting rich entity blocks, but produces more context-efficient summaries for the LLM.
- Both modes respect `max_entities` parameter (default: 30) to cap output size.

## 8. Summary Caching

When `KG_ENABLE_SUMMARY_CACHE` is `True` (default), the KG summary is computed once per turn and cached, avoiding redundant graph traversals during the same `process_turn()` call.

---
*Related: [entity-importance.md](entity-importance.md) for importance scoring details.*
