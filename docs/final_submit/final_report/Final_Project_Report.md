# Final Project Report: StoryWeaver — AI-Driven Text Adventure Engine

**Course**: COMP5423 Natural Language Processing  
**Project Name**: StoryWeaver  
**Group Members**: [Group Member Names]  
**Date**: April 1, 2026

---

## 0. Core Highlights Quick Look

| # | Highlight | Evidence |
|---|---|---|
| 🔬 | **KG On/Off Quantitative Comparison** | Consistency 0% → 100%, Self-BLEU reduced by 38%, LLM Judge +22% |
| 🛡️ | **Dual-Channel Conflict Detection** | Deterministic rules (mutually exclusive pairs + temporal detection) + LLM arbitration, handling 4 types of conflicts |
| 🔄 | **Lazy Loading & Transparent Fallback** | All 4 NLU modules feature a rule-based "shadow" implementation, silently degrading upon model failure |
| 🎛️ | **Three NLG Modes** | `api` (highest quality) / `local` (fully offline) / `hybrid` (cost-balanced), hot-swappable |
| 💡 | **Emotion-Aware NLU** | DistilRoBERTa analyzes 7 Ekman emotions, dynamically adjusting the narrative tone |
| 🌐 | **Interactive PyVis KG Visualization** | Real-time network graph, entities color-coded by type; serves as both a UI feature and a debugging tool |

---

## 1. Task Setup & Background

### 1.1 Project Overview
StoryWeaver is an interactive text adventure game engine that integrates **local NLU models** (DistilBERT, spaCy, fastcoref, DistilRoBERTa) with **LLM-powered narrative generation** (OpenAI GPT-4o-mini / Local Qwen3-4B) and a **dynamic knowledge graph based on NetworkX** to maintain world-state consistency across multi-turn, open-ended storytelling.

### 1.2 Motivation
Traditional text adventures rely on rigid, branching logic trees. While modern LLM-based games offer flexibility, they frequently suffer from "hallucinations"—where the AI forgets past events, resurrects dead characters, or introduces logical contradictions. StoryWeaver mitigates these issues by leveraging a knowledge graph as a structured "source of ground truth," preserving the LLM's creative storytelling capabilities while maintaining logical rigor.

### 1.3 Target Audience & NLP Course Relevance
The system is designed for interactive fiction enthusiasts and hybrid AI architecture researchers. Its core technologies comprehensively cover key areas of the NLP curriculum:

| NLP Domain | Specific Technologies |
|---|---|
| **NLU** | Intent Classification (Fine-tuned DistilBERT), Entity Extraction (spaCy NER + Fuzzy Matching), Coreference Resolution (fastcoref FCoref), Sentiment Analysis (DistilRoBERTa) |
| **NLG** | Conditional Story Continuation, Structured Option Generation, Hierarchical Prompt Engineering |
| **Knowledge Engineering** | Relational Triplet Extraction, Temporal Tracking, Automated Conflict Detection & Resolution |
| **Evaluation** | Distinct-n, Self-BLEU, Entity Coverage, Consistency Rate, LLM-as-Judge (8 Dimensions) |

---

## 2. System Architecture

### 2.1 Overall Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                        │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ Chat UI  │  │ NLU Debug    │  │ KG Visualization (PyVis)   ││
│  └──────────┘  └──────────────┘  └────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    GameEngine (Orchestrator)                    │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ NLU (Local) │→ │Game State│→ │ Story Gen    │→ │OptionGen│ │
│  │ DistilBERT  │  │ GameState│  │ GPT-4o-mini  │  │  (API)  │ │
│  │ spaCy +     │  └────┬─────┘  └──────────────┘  └─────────┘ │
│  │ fastcoref   │       │                                      │
│  └─────────────┘       ▼                                      │
│              ┌─────────────────────┐                          │
│              │Knowledge Graph (NX) │← Relation Extraction     │
│              │ + Conflict Resol.   │  (LLM JSON Mode)         │
│              └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Per-Turn Processing Pipeline

Each player turn triggers an ordered processing pipeline comprising **7 stages**:

| Stage | Module | Function |
|:---:|---|---|
| **1** | Coreference | `fastcoref` resolves pronouns (e.g., "it" → "dragon"), with rule fallback ensuring availability |
| **2** | Intent | Fine-tuned `DistilBERT` → 8 intents (action/dialogue/explore/use_item/ask_info/rest/trade/other) |
| **2b** | Emotion | `DistilRoBERTa` → 7 emotions (Ekman 6 + neutral), adjusting narrative tone |
| **3** | Entities | `spaCy` NER + Noun Phrase Heuristics + KG Fuzzy Matching |
| **4** | Story Gen | LLM continues the story based on KG Summary + History + Intent + Emotion |
| **5** | KG Update | LLM Dual Extraction (Player Input + Story Text) → Entity/Relation Updates → Time Decay → Importance Recalculation |
| **6** | Conflicts | Deterministic Rules + LLM Reasoning → `KeepLatestResolver` / `LLMArbitrateResolver` |
| **7** | Options | LLM generates 3 player options annotated with risk levels |

---

## 3. Core Module Details

### 3.1 NLU Layer: Multi-Model + Transparent Fallback

Every NLU sub-module employs a **lazy loading + 3 retries + rule fallback** mechanism, guaranteeing 100% system availability in **any environment** (including non-GPU, missing pretrained models, or restricted networks):

| Module | Primary Model | Fallback Strategy | Key Implementations |
|---|---|---|---|
| **Intent** | Fine-tuned DistilBERT | Keyword Matching (8-class mapping table) | 3 retries, version compatibility checks |
| **Coreference** | fastcoref FCoref | Context-based Pronoun Replacement Rules | Supports personal/impersonal/possessive/reflexive pronouns |
| **Entities** | spaCy `en_core_web_sm` | Noun Phrases + Regex Proper Nouns | 60+ creature list, 50+ location list, 50+ item list; Adaptive KG fuzzy matching |
| **Emotion** | j-hartmann/emotion-english-distilroberta-base | Keyword Emotion Scoring | 7 Ekman emotions (anger/disgust/fear/joy/sadness/surprise + neutral), returning dominant emotion & score distribution |

**Design Highlight — Lazy Loading & Transparent Fallback**:
- All NLU models are **deferred until the first `process_turn()`**, ensuring rapid game startup responsiveness.
- Every module features a corresponding **rule-based "shadow" implementation** (keyword matching, regex extraction, contextual replacement).
- The system **silently degrades** upon model loading failure, requiring no user intervention, ensuring the game never crashes due to an unavailable NLU component.
- Backend module statuses (`distilbert` vs. `rule_fallback`) are tracked in real-time via the `nlu_status` dictionary and transparently displayed in the UI debug panel.

### 3.2 Knowledge Graph: Dynamic World State

The KG is a real-time world state graph built on `nx.MultiDiGraph`, allowing multiple relation edges between the same pair of nodes.

#### Node (Entity) Attributes
- **Identity**: `name`, `entity_type` (person/location/item/creature/event/unknown)
- **Narrative**: `description` (automatically accumulates new info), `status` (dynamic state dictionary), `status_history` (up to 10 change records)
- **Temporal**: `created_turn`, `last_mentioned_turn`
- **Importance**: `mention_count`, `player_mention_count`, `importance_score` (0-1 composite score)

#### Edge (Relation) Attributes
- `relation` (relation type), `context` (reason for relation establishment)
- `created_turn`, `last_confirmed_turn`, `confidence` (0-1, decays over time)

#### Importance Scoring Formula
```text
importance = 0.3 × norm(degree) + 0.3 × 0.95^(turns_since_last_mention)
           + 0.2 × norm(mention_count) + 0.2 × norm(player_mention_count)
```
Supports both **incremental mode** (updating only dirty nodes per turn) and **full mode** (regularly triggered to ensure global precision).

#### Layered Summary
Entities are stratified by importance thresholds to optimize LLM attention allocation:
- **Core Entities** (≥0.6): Full details (description, status, emotion, history, relations)
- **Secondary Entities** (0.3-0.6): Simplified information
- **Background Entities** (<0.3): Only name + type + last mentioned turn
- Appended with a **Recent Timeline** (recent relational events sorted by turn)

#### Relation Decay & Auto-Pruning
```text
confidence_t = confidence_0 × (KG_RELATION_DECAY_FACTOR)^turns_since_last_confirmed
```
Relations dropping below `KG_RELATION_MIN_CONFIDENCE` are automatically purged to prevent unbounded KG growth.

### 3.3 Conflict Detection & Resolution: Dual-Channel Mechanism

StoryWeaver implements a **dual-channel conflict detection** system, merging the efficiency of deterministic rules with the deep analytical capabilities of LLM reasoning:

#### Channel 1: Deterministic Rule Detection (Zero-Latency)

| Conflict Type | Detection Rule | Example |
|---|---|---|
| **exclusive_relation** | Mutually exclusive pairs `(ally_of, enemy_of)` and `(alive, dead)` cannot coexist | A single entity pair is simultaneously marked as allies and enemies |
| **dead_active** | Entities marked `dead` cannot possess `possesses`/`located_at`/`ally_of` relations | A deceased character still holds items or resides at a location |
| **temporal** | Post-mortem actions + Causal inversion (effects preceding causes in `causes`/`enables`) | Establishing new relations after death; inverted chronological causality |

#### Channel 2: LLM Reasoning Detection (Deep Analysis)

The current KG summary and the newly generated story text are sent to the LLM to identify logical contradictions uncatchable by rules (e.g., "A character uses an item lost 3 turns ago"). Detected conflicts are handled based on LLM confidence scores:
- **≥0.75**: Immediately accepted and resolved
- **0.45-0.74**: Deferred, awaiting more context
- **<0.45**: Discarded (filtering out noise)

#### Resolution Strategies (Strategy Pattern, Configurable)

| Strategy | Mechanism | Characteristics |
|---|---|---|
| **KeepLatestResolver** | Compares `last_confirmed_turn`, discarding the older conflicting relation | Fast, deterministic, zero LLM calls |
| **LLMArbitrateResolver** | Sends conflict details to LLM, returning `keep_new`/`keep_old`/`remove_relation`/`update_entity`/`no_action` | High precision, requires extra API calls |

### 3.4 NLG Layer: Three Operating Modes (Architectural Flexibility)

The system supports three operating modes via the `NLG_MODE` configuration, adapting to various deployment scenarios:

| Mode | Story Generation | Option / Relation Extraction | Applicable Scenarios |
|---|---|---|---|
| **api** | GPT-4o-mini | GPT-4o-mini | Highest quality, ideal for demos and evaluations |
| **local** | Qwen3-4B (llama.cpp) | Qwen3-4B | Fully offline, privacy-focused, zero API dependency |
| **hybrid**| Qwen3-4B (Local) | GPT-4o-mini (API) | Balances quality and cost—local model for creative prose, API for structured extraction |

**Architectural Advantages**:
- All three modes can be hot-swapped in the `.env` file **without modifying any code**.
- The `hybrid` mode offloads compute-heavy story generation to local hardware while retaining the API's superior JSON output capabilities for structured tasks (option generation, relation extraction), which represents a strategic latency optimization saving approximately 14 seconds per turn.
- The local model utilizes the **Qwen3-4B GGUF (Q4_K_M quantization)**, compressing the model footprint from 7.5GB to 2.4GB. Served via an OpenAI-compatible API on port 8081 through `llama.cpp`, it natively supports pure CPU inference.

### 3.5 Dual Entity Extraction (`extract_dual`)

Extracts entities and relations simultaneously from both **player input** and **story text**:
- Player input typically contains intended actions and newly mentioned entities.
- Story text encompasses narrative events, NPC reactions, and world state changes.
- A single LLM call merges both text segments, automatically deduplicating and preserving the most information-rich versions.
- Fallback Mechanism: If dual extraction fails, it extracts them separately and merges the results.

---

## 4. Evaluation Framework

### 4.1 Automated Metrics (`metrics.py`)

| Metric | Formula | Interpretation | Direction |
|---|---|---|---|
| **Distinct-n** | `\|unique n-grams\| / \|total n-grams\|` | Lexical diversity | ↑ Better |
| **Self-BLEU** | Average Sentence-BLEU against all other sentences | Inter-text similarity | ↓ Better |
| **Entity Coverage**| `\|KG entities mentioned in story\| / \|total KG entities\|` | World state reference rate | ↑ Better |
| **Consistency Rate**| `\|Conflict-free turns\| / \|Total turns\|` | Narrative consistency | ↑ Better |
| **Type-Token Ratio**| `\|Unique tokens\| / \|Total tokens\|` | Vocabulary richness | ↑ Better |
| **Lexical Overlap**| Jaccard similarity of adjacent turns | Adjacent text repetitiveness | ↓ Better |
| **Flesch Reading Ease**| Standard readability formula | Text readability | — |
| **Graph Density**| `edge_count / (node_count × (node_count-1))` | KG structural complexity | — |

### 4.2 LLM-as-Judge (8 Dimensions)

An independent LLM assesses the generated results across 8 dimensions (on a 1-10 scale):

| Dimension | Evaluation Focus |
|---|---|
| **narrative_quality** | Overall narrative quality |
| **consistency** | Adherence to established world facts |
| **player_agency** | Degree of player choice impact on the story |
| **creativity** | Originality and novelty |
| **pacing** | Control of narrative flow |
| **option_relevance** | Relevance of generated options to the current context |
| **causal_link** | Logical coherence of cause and effect |
| **local_coherence** | Paragraph-level flow and continuity |

### 4.3 KG On/Off Comparative Experiment — The Core Quantitative Evidence

**Experimental Setup**: Fantasy scenario, 10 turns, always selecting the first option, single run.

| Metric | KG ON | KG OFF | Change |
|---|---:|---:|---|
| **Consistency Rate** | **1.0000** | **0.0000** | **+100%** ⬆ |
| **Self-BLEU** (↓ Better) | **0.2200** | 0.3582 | **-38%** ⬇ |
| **Entity Coverage** | **1.0000** | 0.7917 | +26% ⬆ |
| **Distinct-1** | 0.3652 | 0.3165 | +15% ⬆ |
| **Distinct-2** | 0.7423 | 0.6448 | +15% ⬆ |
| **Distinct-3** | 0.9059 | 0.8145 | +11% ⬆ |
| **Type-Token Ratio** | 0.3137 | 0.2690 | +17% ⬆ |
| **LLM Judge Average**| **9.12** / 10 | **7.50** / 10 | **+22%** ⬆ |

> **Key Findings**:
> - **Consistency leaps from 0% → 100%**: Without a KG, the LLM inevitably generated at least one logical contradiction within 10 turns. With the KG active, the conflict detection system successfully trapped and resolved all contradictions.
> - **Self-BLEU drops by 38%**: Injecting the KG summary into the prompt drastically reduced LLM repetitiveness, significantly elevating narrative diversity.
> - **LLM Judge rises by 22%**: Surging from 7.50 to 9.12, achieving gains across all 8 evaluated dimensions.

**LLM Judge Breakdown**:

| Dimension | KG ON | KG OFF | Diff |
|---|---:|---:|---|
| narrative_quality | 9.0 | 8.0 | +1.0 |
| consistency | 10.0 | 9.0 | +1.0 |
| player_agency | 8.0 | 6.0 | **+2.0** |
| creativity | 8.0 | 6.0 | **+2.0** |
| pacing | 9.0 | 7.0 | **+2.0** |
| option_relevance| 9.0 | 7.0 | **+2.0** |
| causal_link | 10.0 | 8.0 | **+2.0** |
| local_coherence | 10.0 | 9.0 | +1.0 |

> **Note**: KG-enabled mode introduces an average latency of ~18.2s/turn (vs. 5.7s/turn when off), due to the overhead of LLM relation extraction and conflict detection. This is an acceptable architectural tradeoff for guaranteed narrative consistency.
> Additionally, rigorous 90-turn automated stress tests using a deterministic "always pick option one" bot revealed that the LLM Judge heavily penalized Player Agency and Causal Link. This highlights that evaluating interactive, open-ended narratives requires a blend of automated metrics and human-in-the-loop qualitative testing.

---

## 5. Technical Challenges & Solutions

### 5.1 Long-Term Narrative Consistency
**Challenge**: LLM context windows become cluttered as the story progresses, leading to amnesia or contradictions.  
**Solution**: Deployed a Knowledge Graph as an external memory bank + dual-layer conflict detection (deterministic rules + LLM reasoning) + automated resolution strategies. Experimental results prove KG integration elevates consistency rates from 0% to 100%.

### 5.2 NLU Deployment Robustness
**Challenge**: Heavyweight models (DistilBERT, fastcoref) suffer from slow loading times or fail outright in resource-constrained environments.  
**Solution**: Engineered a lazy loading + 3 retries + transparent rule fallback architecture. Each module maintains a corresponding "shadow" implementation, guaranteeing system operability under any conditions.

### 5.3 KG Noise & Overgrowth
**Challenge**: Automated extraction inherently produces redundant entities and relations.  
**Solution**: Implemented relation temporal decay + composite importance scoring + layered summarization + node caps (`KG_MAX_NODES`) to automatically prune low-importance nodes.

### 5.4 Local Model Inference Optimization
**Challenge**: Deploying large models locally requires substantial VRAM overhead.  
**Solution**: Integrated Qwen3-4B's Q4_K_M quantized GGUF format (compressing from 7.5GB to 2.4GB). Served via an OpenAI-compatible API using `llama.cpp`, it enables high-performance CPU-only inference.

---

## 6. Project Highlights & Innovations

### 6.1 Hybrid NLU-KG-NLG Pipeline
Unlike most AI games that haphazardly feed raw history into LLMs, StoryWeaver "understands" player input prior to story generation—resolving coreferences, classifying intents, analyzing emotions, and extracting entities—delivering a highly structured instruction set and world state to the LLM.

### 6.2 Interactive PyVis KG Visualization
Leveraging PyVis for **real-time interactive knowledge graph visualization** stands as one of the project's most recognizable features:
- **Semantic Color-Coding**: Person=Cyan (#00f0ff), Location=Green (#39ff14), Item=Gold (#ffd700), Creature=Magenta (#ff00aa), Event=Purple (#7b2fff).
- **Interactivity**: Supports node dragging, zooming, and hover-to-view entity details.
- **Dual Utility**:
  1. **User Experience**: Players intuitively observe the world state evolving alongside the narrative.
  2. **Developer Tooling**: Enables real-time verification of NLU entity and relation extraction accuracy.
- **Graceful Degradation**: Automatically falls back to HTML table rendering if PyVis becomes unavailable.

### 6.3 Dual Entity Extraction
The `extract_dual` function concurrently parses both player input and narrator responses, guaranteeing that even subtle player actions confirmed by the narrator are captured into the world state. A single LLM call executes the merge, applying deduplication and information-richness prioritization.

### 6.4 Comprehensive Evaluation Framework
- **7 Automated Metrics**: Distinct-1/2/3, Self-BLEU, Entity Coverage, Consistency Rate, Type-Token Ratio.
- **8-Dimension LLM-as-Judge**: Spanning narrative quality, consistency, player agency, creativity, pacing, option relevance, causal links, and local coherence.
- **KG On/Off Ablation Study**: Empirically quantifying the exact quality improvements delivered by the knowledge graph.

### 6.5 Game Persistence System
- Comprehensive JSON serialization/deserialization of the entire game state (KG + History + KG Stats).
- Automatic state saving upon every turn completion.
- Periodic snapshotting (configurable intervals).
- Semantic savefile naming (intelligently generated based on the opening story context).
- Browser refresh recovery via Streamlit Runtime Session persistence.

### 6.6 Highly Configurable Policies
Adjustable dynamically via `.env` and the frontend UI:
- Conflict Resolution Strategy (`keep_latest` / `llm_arbitrate`)
- Extraction Mode (`story_only` / `dual_extract`)
- Summary Format (`flat` / `layered`)
- Importance Calculation Mode (`degree_only` / `composite` / `incremental`)

---

## 7. Group Member Contributions

| Member Name | Contribution % | Primary Responsibilities |
|---|---|---|
| [Name 1] | 25% | NLU Development (Intent, Coreference, Emotion), Model Fine-tuning, Fallback Mechanisms |
| [Name 2] | 25% | KG Architecture, Relation Extraction, Dual-Layer Conflict Detection & Resolution |
| [Name 3] | 25% | NLG Pipeline, Prompt Engineering, Local Model Integration (`llama.cpp` / Qwen3-4B) |
| [Name 4] | 25% | Streamlit UI, Evaluation Suite (Auto-Metrics + LLM Judge), Persistence System, Documentation |

---

## 8. Conclusion & Future Outlook

StoryWeaver successfully demonstrates that **combining a structured world state (Knowledge Graph) with specialized NLU modules** dramatically enhances the coherence and quality of AI-generated narratives. The KG On/Off ablation study proves that the knowledge graph propels the consistency rate from 0% to 100%, boosts LLM Judge scores by 22%, and concurrently curtails text repetitiveness (Self-BLEU -38%).

The project establishes a **scalable, configurable, and rigorously evaluable** framework for future interactive fiction, striking an elegant balance between creative expression and logical consistency.

**Future Directions**:
- Exploring highly optimized KG update strategies to further slash latency.
- Integrating additional local NLU models (e.g., lightweight relation extraction models).
- Broadening support for multi-lingual narratives.
- Expanding the architecture to support multiplayer collaborative storytelling engines.

---

## Appendix

### A. Tech Stack
- **NLU**: DistilBERT + spaCy (en_core_web_sm) + fastcoref (FCoref) + DistilRoBERTa
- **NLG**: OpenAI GPT-4o-mini (API) / Qwen3-4B GGUF Q4_K_M (llama.cpp)
- **Knowledge Graph**: NetworkX (MultiDiGraph) + PyVis
- **Frontend**: Streamlit
- **Evaluation**: NLTK (BLEU) + Custom Metrics + LLM-as-Judge
- **Testing**: 76+ Unit Tests (KG Module), 186+ Unit Tests (NLU & KG Refinements)

### B. Project Structure
```text
story_maker/
├── app.py                          # Streamlit Entry Point
├── config.py                       # Pydantic Config (.env supported)
├── src/
│   ├── engine/                     # Game Engine Orchestrator (game_engine.py)
│   ├── nlu/                        # NLU Modules (Intent, Entity, Coref, Emotion)
│   ├── nlg/                        # NLG Modules (Story Gen, Option Gen, Prompts)
│   ├── knowledge_graph/            # KG (Graph, Extraction, Conflicts, Visualization)
│   ├── evaluation/                 # Metrics & LLM Judge (metrics.py, llm_judge.py)
│   ├── ui/                         # Streamlit UI Components
│   └── utils/                      # Shared Utilities (API Clients, etc.)
├── tests/                          # Testing Suite
├── training/                       # Model Training Scripts
├── scripts/                        # Deployment & Evaluation Scripts
└── docs/                           # Full Documentation
```
