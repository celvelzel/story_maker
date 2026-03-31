# Entity Importance Strategy

This document details the Knowledge Graph (KG) entity pruning and importance scoring mechanism in the `story_weaver` project. This feature is managed via the `KG_IMPORTANCE_MODE` parameter in the configuration.

## 1. Functional Overview

In long-form narratives, the Knowledge Graph grows as the number of turns increases. To ensure the effectiveness of the LLM context, the system identifies which entities are core to the current story and which are outdated background information. The entity pruning strategy determines an entity's `importance_score` (0.0 to 1.0), which affects its priority during context window construction and summary generation.

## 2. Strategy Comparison

Two modes are currently supported: `composite` (default) and `degree_only`.

### 1. Composite (Recommended)
This is the system's default intelligent mode, combining graph structure, temporal decay, and player interaction.

**Calculation Formula:**
`Importance = 0.3 * norm(degree) + 0.3 * recency + 0.2 * norm(mention_count) + 0.2 * norm(player_mention_count)`

*   **norm(degree)**: Normalized degree of the node (more connections result in a higher base weight).
*   **recency**: Uses an exponential decay formula `0.95 ^ turns_since_last_mention`. The longer it has been since the last mention, the faster the weight drops.
*   **mention_count**: Total frequency of the entity appearing in the story text.
*   **player_mention_count**: Number of times the player directly mentions the entity in their input.

**Advantages:**
- **Dynamic Pruning**: Characters who haven't appeared for a long time (e.g., prologue characters) naturally lose importance.
- **Player-Oriented**: Items or characters repeatedly mentioned by the player receive a significant weight boost.
- **Context-Friendly**: Ensures that limited context space is always reserved for the most active entities.

### 2. Degree Only
A traditional graph theory evaluation method, primarily for backward compatibility or minimalist scenarios.

**Core Logic:**
The entity's weight depends entirely on the number of edges it has in the graph.

**Limitations:**
- **Inability to Prune**: A "dead" character who established many relationships early on will keep a high score forever.
- **Lack of Timeliness**: Cannot distinguish between the current focus and historical background.

## 3. Configuration Parameters

You can fine-tune the following parameters in the `.env` file to change pruning behavior:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `KG_IMPORTANCE_MODE` | `composite` | Switch mode: `composite` or `degree_only` |
| `KG_IMPORTANCE_DECAY_FACTOR` | `0.95` | Importance decay factor for each turn without a mention |
| `KG_IMPORTANCE_MENTION_BOOST` | `0.15` | Importance boost for each mention in the story |
| `KG_IMPORTANCE_PLAYER_BOOST` | `0.3` | Extra weight bonus when directly mentioned by the player |

## 4. How to Switch

### 1. Configuration File
Modify in `.env`:
```env
KG_IMPORTANCE_MODE=composite
```

### 2. Frontend Interface
In the Streamlit sidebar **"⚙ KG Strategy Settings"** panel, select the **"Entity Pruning Strategy"** dropdown.

### 3. Code Invocation
```python
from src.engine.game_engine import GameEngine
engine = GameEngine(importance_mode="composite")
```

## 5. Implementation Details

The importance scoring is calculated in the `KnowledgeGraphManager` during each `update_graph` call. Entities with scores below a certain threshold (default 0.1) may be excluded from the immediate LLM context window to save tokens, while still being preserved in the full graph for potential future recall.

---
*Related Reference: For detailed technical implementation, see [docs/reports/kg-optimization.md](../reports/kg-optimization.md)*
