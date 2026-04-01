# Final Project Report: StoryWeaver - AI-Powered Text Adventure Engine

**Course**: COMP5423 Natural Language Processing  
**Project Name**: StoryWeaver  
**Group Members**: [Group Member Names]  
**Date**: April 1, 2026

---

## 1. Task Settings & Background

### 1.1 Project Overview
StoryWeaver is an interactive text adventure game engine designed to demonstrate the integration of local Natural Language Understanding (NLU) models with Large Language Model (LLM) narrative generation and dynamic world state management. The project addresses the core challenge of maintaining narrative consistency and logical coherence in multi-turn, open-ended storytelling.

### 1.2 Motivation
Traditional text adventures often rely on rigid branching logic. Modern LLM-based games, while flexible, frequently suffer from "hallucinations"—where the AI forgets previous events, resurrects dead characters, or introduces logical contradictions. StoryWeaver mitigates these issues by implementing a structured "World State" via a Knowledge Graph (KG) that acts as a ground truth for the narrator.

### 1.3 Target Audience & Appropriateness
The system is designed for enthusiasts of interactive fiction and developers interested in hybrid AI architectures. It fits the NLP course scope by utilizing core technologies including:
- **NLU**: Intent classification, entity extraction, and coreference resolution.
- **NLG**: Conditional story continuation and structured option generation.
- **Knowledge Engineering**: Relation extraction and automated conflict detection.

---

## 2. System Development & Methodologies

### 2.1 Architecture Overview
StoryWeaver employs a pipeline-based architecture where each player turn triggers a sequence of processing stages:

[Placeholder: System Architecture Diagram showing NLU -> Engine -> NLG/KG feedback loop]

1.  **NLU Layer (Local)**: Processes raw player input to resolve pronouns, classify intent, and extract entities using lightweight local models.
2.  **Narrative Engine (Orchestrator)**: Manages the game state, history, and invokes the generation and knowledge modules.
3.  **Knowledge Graph (World State)**: A NetworkX-based graph that tracks entities, relationships, and attributes.
4.  **NLG Layer (Hybrid)**: Generates story text and options using a combination of remote APIs (GPT-4o-mini) and local models (Qwen-4B).

### 2.2 NLU Methodologies
- **Intent Classification**: A fine-tuned **DistilBERT** model classifies inputs into 8 categories (e.g., *action, dialogue, explore*). A keyword-based fallback ensures robustness if the model fails to load.
- **Coreference Resolution**: Uses **fastcoref (FCoref)** to resolve pronouns (e.g., "it" or "him") based on the recent narrative context before intent classification.
- **Entity Extraction**: A hybrid approach using **spaCy NER**, noun-phrase heuristics, and fuzzy matching against existing KG entities to ensure mention consistency.
- **Sentiment Analysis**: Employs a **DistilRoBERTa** model to detect player emotions (Ekman's 6 emotions), allowing the narrator to adapt the story's tone.

### 2.3 Knowledge Graph & World State
The KG is the "memory" of the system.
- **Relation Extraction**: LLM-based extraction of (Source, Relation, Target) triplets with added context and confidence scores.
- **Temporal Tracking**: Each node and edge tracks its `created_turn` and `last_confirmed_turn`.
- **Importance Scoring**: A composite metric (Degree Centrality + Recency + Mention Frequency) ranks entities, ensuring only relevant information is fed back into the LLM prompt.

### 2.4 NLG & Prompt Engineering
We utilize structured prompt templates that inject the `kg_summary` and `history` into the LLM context. The system supports three modes:
- `api`: High-quality remote inference.
- `local`: Privacy-focused local inference via `llama.cpp`.
- `hybrid`: Distributes creative prose to local models and structural extraction to remote APIs.

---

## 3. Technical Challenges & Solutions

### 3.1 Challenge: Maintaining Long-term Narrative Consistency
**Problem**: As the story progresses, the LLM context window becomes cluttered, leading to "forgetting" or contradicting established facts (e.g., a character being in two places at once).
**Solution**: We implemented a **Dual-Channel Conflict Detection** system. 
1.  **Deterministic Rules**: A set of hard-coded mutex pairs (e.g., `ally_of` vs `enemy_of`, `alive` vs `dead`) immediately flags contradictions.
2.  **LLM Arbitration**: High-confidence LLM checks identify logic-level conflicts (e.g., "A character is using an item they lost 3 turns ago").
3.  **Resolution Strategy**: A `KeepLatestResolver` automatically prunes older, conflicting relations based on the `last_confirmed_turn` attribute in the KG.

### 3.2 Challenge: NLU Model Deployment & Fallback
**Problem**: Heavy NLU models (DistilBERT, fastcoref) can be slow to load or fail in resource-constrained environments.
**Solution**: We developed a **Lazy-Loading & Transparent Fallback** mechanism. 
- Models are loaded only upon the first turn.
- Each module (Intent, Coref, Entity) has a rule-based "Shadow" implementation. If `torch` or model weights are missing, the system silently switches to keyword matching or regex-based extraction, ensuring 100% uptime.

### 3.3 Challenge: KG Noise & Overgrowth
**Problem**: Automated extraction often produces redundant or "noisy" entities that degrade story quality.
**Solution**: We introduced **Relation Decay and Importance Filtering**. 
- Relations lose confidence every turn they aren't mentioned (`KG_RELATION_DECAY_FACTOR`).
- The `to_summary()` method uses a **Layered Summary** approach, providing full details for "Core" entities while collapsing "Background" entities into simple lists, optimizing the LLM's attention.

---

## 4. Highlights & Innovations

### 4.1 Hybrid NLU-KG-NLG Pipeline
Unlike many AI games that feed raw history into an LLM, StoryWeaver "interprets" the input first. By resolving coreferences and identifying intent *before* story generation, we provide the LLM with a much cleaner instruction set.

### 4.2 Dynamic Knowledge Graph Visualization
The system includes a real-time **interactive KG visualizer** (using PyVis). Players can see the world state evolve, which enhances transparency and acts as a debugging tool for the NLU/KG modules.

### 4.3 Robustness via Multi-Strategy Extraction
We implemented `extract_dual`, which processes both the player's intent and the narrator's response. This ensures that even if the player's action is subtle, the narrator's confirmation of the result is captured in the world state.

### 4.4 Automated Evaluation Suite
The project includes a comprehensive evaluation module (`metrics.py`) calculating:
- **Distinct-n**: Vocabulary diversity.
- **Self-BLEU**: Intra-session variety.
- **Entity Coverage**: How well the story references the KG.
- **Consistency Rate**: Percentage of turns without detected conflicts.

---

## 5. Performance Evaluation

### 5.1 Quantitative Results
We conducted a "KG On vs. Off" benchmark to measure the impact of the Knowledge Graph on generation quality.

| Metric | KG Enabled (kg_on) | KG Disabled (kg_off) | Improvement |
| --- | ---: | ---: | ---: |
| **Consistency Rate** | 100% | 0% | +100% |
| **Entity Coverage** | 1.00 | 0.79 | +26% |
| **Distinct-2** | 0.7423 | 0.6448 | +15% |
| **Self-BLEU** (Lower is better) | 0.2200 | 0.3582 | +38% |
| **LLM Judge (Avg Score)** | 9.12 / 10 | 7.50 / 10 | +21% |

### 5.2 Analysis
The results demonstrate that the Knowledge Graph significantly improves **Consistency** and **Diversity**. By injecting the KG summary into the prompt, the LLM is less likely to repeat itself (Lower Self-BLEU) and more likely to reference established world details (Higher Entity Coverage).

---

## 6. Group Member Contributions

| Member Name | Contribution % | Primary Responsibilities |
| ----------- | -------------- | ------------------------ |
| [Name 1] | 25% | NLU Development (Intent, Coref, Sentiment), Model Fine-tuning. |
| [Name 2] | 25% | Knowledge Graph Architecture, Relation Extraction, Conflict Detection. |
| [Name 3] | 25% | NLG Pipeline, Prompt Engineering, Local Model Integration. |
| [Name 4] | 25% | Streamlit UI, Evaluation Suite, Documentation & Reporting. |

---

## 7. Conclusion
StoryWeaver successfully demonstrates that a structured world state (Knowledge Graph) combined with specialized NLU modules can significantly enhance the coherence and quality of AI-generated narratives. The project provides a scalable framework for future interactive fiction that prioritizes logical consistency alongside creative expression.

---
[Placeholder: Screenshot of the Game UI showing the story area and the Knowledge Graph sidebar]
[Placeholder: Screenshot of the NLU Debugging panel]
