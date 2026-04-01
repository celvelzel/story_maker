# StoryWeaver Presentation Script (Final Version)

This document contains the presentation script and speaker notes for the COMP5423 NLP group project.
Target Duration: 7-8 minutes (approx. 1,000-1,100 words).

> **Design Principle**: Every slide highlights the unique technical value of the project. All technical details are verified by code to ensure accuracy.

---

## Slide 1: Title Page

**Visual Suggestion:** A minimalist, dark tech-style title page with the "StoryWeaver" logo. The background features a glowing, interconnected knowledge graph network. Include group members' names, student IDs, and the course code.

**Slide Content:**
* StoryWeaver: AI-Driven Text Adventure Game Engine
* Dynamic Narrative Generation via Hybrid NLU + Knowledge Graph + LLM
* COMP5423 NLP Group Project
* Group [Your Group ID]

**Speaker Notes (approx. 30 seconds):**
Good evening, everyone. We are Group [Your Group ID], and today we present StoryWeaver—an AI-driven text adventure game engine. Our core objective was to solve a critical challenge: how to keep AI creative while maintaining logical consistency in an open-ended, multi-turn narrative. To achieve this, we combined local NLU models, a dynamic Knowledge Graph, and Large Language Models to build a complete interactive narrative system.

---

## Slide 2: Motivation & Problem Statement

**Visual Suggestion:** Split-screen comparison. Left: Traditional text games (rigid decision trees, "I don't understand" errors, limited options). Right: StoryWeaver (dynamic branching, natural language input, infinite possibilities). A red arrow in the middle highlights the "Hallucination Problem".

**Slide Content:**
* **Limitations of Traditional Text Adventures:**
  * Rigid, hard-coded decision trees
  * Player input outside presets → "I don't understand"
  * AI forgets early plot points in long sessions (Hallucination)
* **StoryWeaver's Breakthrough:**
  * Natural language input, infinite dynamic plots
  * Knowledge Graph anchors world state, eliminating contradictions
  * Hybrid Architecture: Local NLU + Cloud/Local LLM

**Speaker Notes (approx. 50 seconds):**
Traditional text adventure games rely on hard-coded decision trees. Players can only choose from preset options, and if the input falls outside the script, the game simply says "I don't understand." Modern LLM-based games, while flexible, face an even more severe problem: the AI forgets early events, resurrects dead characters, or even places a character in two locations at once. This is known as the "hallucination" problem.

StoryWeaver's solution is to use a structured Knowledge Graph as the "ground truth source." 
**Take an extreme example**: If the player kills a goblin in Turn 2, but in Turn 10 the LLM tries to generate "The goblin waves at you." In our system, this logical contradiction is immediately caught and blocked by the Knowledge Graph. We don't just dump the raw chat history into the LLM; instead, we first use a local NLU model to understand the intent, and then use the graph's structured state to constrain the generation. This preserves the LLM's creativity while ensuring absolute narrative consistency.

---

## Slide 3: System Architecture

**Visual Suggestion:** Clear high-level architecture diagram (consistent with README). Data flow from left to right: Player Input → NLU Pipeline → Game Engine Orchestrator → Story/Option Generation (LLM) ↔ Knowledge Graph. Top marked as Streamlit Frontend. Use different colors to distinguish local modules (green) and API modules (blue).

**Slide Content:**
* **Frontend:** Streamlit UI — Chat Interface + Interactive PyVis KG Visualization + NLU Debug Panel
* **NLU (Local):** DistilBERT (Intent Classification) + spaCy (Entity Extraction) + fastcoref (Coreference Resolution) + DistilRoBERTa (Emotion Analysis)
* **NLG (Three Modes):** API (Mimo v2 Flash) / Local (Qwen3-4B via llama.cpp) / Hybrid
* **World State:** NetworkX MultiDiGraph + Dual-Channel Conflict Detection + Importance Scoring
* **Engineering Robustness:** Lazy Loading + Transparent Fallback (Auto-downgrade to rules on model failure)

**Speaker Notes (approx. 60 seconds):**
Here is the overall architecture of StoryWeaver. The system consists of four layers:

At the top is the Streamlit frontend, which not only provides the game chat interface but also includes an interactive PyVis-based knowledge graph visualization—where entities are color-coded (cyan #00f0ff for characters, green #39ff14 for locations)—along with an NLU debug panel to let us observe the AI's underlying decision-making process.

The second layer is the local NLU pipeline, comprising four modules: fastcoref resolves pronoun ambiguity; our fine-tuned DistilBERT classifies player intent into 8 categories; spaCy extracts game entities; and a DistilRoBERTa-based emotion analyzer identifies 7 Ekman emotions. All models run locally, making them fast and cost-free.

The third layer is the NLG generation module. Routed via HybridClientManager, it supports three modes: pure API calls to Mimo v2 Flash, pure local execution using Qwen3-4B, and a hybrid mode where creative story generation runs locally while structured option generation goes through the API.

At the bottom is the Knowledge Graph, built on NetworkX MultiDiGraph, paired with dual-channel conflict detection. The entire system features lazy loading and transparent fallbacks: the load() method of each NLU module retries up to 3 times before automatically downgrading to keyword rule matching, ensuring 100% availability.

---

## Slide 4: NLU Pipeline — Understanding Player Intent

**Visual Suggestion:** Flowchart showing the transformation of an example sentence. "I attack the goblin with it" → fastcoref resolves "it" → "the goblin" → DistilBERT classifies as "attack" → spaCy extracts entity "goblin". Color-code the output of each module.

**Slide Content:**
* **Coreference Resolution (fastcoref):** "I attack it" → "I attack the goblin", fallback to pronoun mapping rules
* **Intent Classification (DistilBERT):** 8 intent categories, fallback using KEYWORD_MAP scoring
* **Entity Extraction (spaCy NER + Heuristics):** Creature/Location/Item/Spell word sets, with fuzzy matching disambiguation
* **Emotion Analysis (DistilRoBERTa):** 7 Ekman emotions (Anger/Disgust/Fear/Joy/Sadness/Surprise/Neutral), injected into generation prompt

**Speaker Notes (approx. 55 seconds):**
Let's dive into the NLU pipeline. When a player types "I attack it with the sword", the system executes four steps sequentially:

First, fastcoref resolves "it" to a specific entity, like "the goblin", based on recent story context. If the model fails to load, the system falls back to predefined pronoun mapping rules.

Next, the resolved text is fed into our fine-tuned DistilBERT model, which classifies the intent into one of 8 categories—in this case, "Attack". If unavailable, the system automatically switches to KEYWORD_MAP scoring.

Then, spaCy's NER, combined with custom noun-phrase heuristics, extracts specific game entities. We predefined vocabularies for creatures, locations, items, and spells, and use fuzzy matching against KG entities for disambiguation.

Finally, the emotion analyzer uses the j-hartmann/emotion-english-distilroberta-base model to identify 7 Ekman emotions. This emotional result is directly injected into the StoryGenerator's continue_story method, allowing the narrator to adjust the story's tone based on the player's mood.

---

## Slide 5: Knowledge Graph — The Core of Narrative Consistency

**Visual Suggestion:** Screenshot of the graph network visualization (interactive PyVis style). Highlight: Node attributes (status, location, relations), "Conflict Detected!" warning box, and the hierarchical summary generation process.

**Slide Content:**
* **Importance Scoring Formula:** 0.3×Degree Centrality + 0.3×Recency(0.95^n) + 0.2×Mention Freq + 0.2×Player Mention
* **Hierarchical Summary:** Core(≥0.6) / Secondary(≥0.3) / Background(<0.3), optimizes LLM context window
* **Dual-Channel Conflict Detection:**
  * Channel 1 — Deterministic Rules: EXCLUSIVE_PAIRS (alive/dead, ally/enemy) + Temporal Causality Check
  * Channel 2 — LLM Arbitration: Confidence Tiers (>0.75 Accept / 0.45-0.74 Delay / <0.45 Ignore)
* **Conflict Resolution:** KeepLatestResolver (retains latest by last_confirmed_turn) + LLMArbitrateResolver

**Speaker Notes (approx. 75 seconds):**
The Knowledge Graph is StoryWeaver's "memory center." Instead of cramming the entire chat history into the LLM, we extract entities and relations after each turn to update the graph.

The graph is implemented using NetworkX MultiDiGraph. We designed an importance scoring system for each entity: Importance = 0.3 × Degree Centrality + 0.3 × Recency (decayed by 0.95 to the power of unmentioned turns) + 0.2 × Mention Frequency + 0.2 × Player Mention Frequency. Unmentioned entities decay over time, and relationship confidence also decreases, with those below a threshold automatically pruned.

Based on these scores, the to_summary() method generates a hierarchical summary: Core entities (≥0.6) provide full attributes, statuses, emotions, and recent history; Secondary entities (≥0.3) offer basic info; Background entities (<0.3) only retain names and types. This provides the LLM with a compact yet precise context.

When new relationships are extracted, dual-channel conflict detection kicks in. Channel One uses deterministic rules: EXCLUSIVE_PAIRS define mutually exclusive states like alive/dead or ally/enemy, alongside "dead-active" checks (dead entities cannot form new relations). Temporal checks identify causal inversions. Channel Two uses LLM arbitration to analyze the world state summary and new text for deep logical contradictions, categorizing them by confidence: >0.75 directly accepted, 0.45 to 0.74 delayed, and <0.45 ignored.

Conflict resolution is handled collaboratively by KeepLatestResolver (automatically retaining the latest relation based on last_confirmed_turn) and LLMArbitrateResolver (querying the LLM for specific actions like retaining, deleting, or updating).

---

## Slide 6: NLG and Dynamic Story Generation

**Visual Suggestion:** Streamlit interface screenshot highlighting the generated story text and three dynamic options (with Risk Level tags: Low/Medium/High). Right side annotating the structure of the prompt template.

**Slide Content:**
* **Context-Aware Generation:** Player Intent + KG Hierarchical Summary + 7 Emotion Tones + History
* **Dynamic Options Generation:** 3 choices per turn, Risk-graded (Low/Medium/High), structured output via llm_client.chat_json()
* **Three NLG Modes:**
  * API: Mimo v2 Flash, highest quality
  * Local: Qwen3-4B via llama.cpp, privacy-first
  * Hybrid: Story→Local, Options/Relation Extraction→API
* **LLM-Agnostic Architecture:** Routed via HybridClientManager, seamless switching

**Speaker Notes (approx. 55 seconds):**
For story generation, we designed a structured prompt template that merges the player's classified intent, emotional tone, the Knowledge Graph's hierarchical summary, and history into a compact context package. This ensures the LLM's narrative continuation strictly adheres to the established world rules.

At the end of each turn, the option generation module uses structured chat_json output to create three distinct choices for the player, categorized by risk level: Low Risk (safe exploration), Medium Risk (standard action), and High Risk (dramatic twist). Even if players don't want to type custom responses, this keeps them highly engaged.

Our architecture is LLM-agnostic—routed through HybridClientManager's get_client_for_task method. You can use the Mimo API for the highest quality, a local Qwen3-4B for privacy, or a hybrid mode where creative story generation uses the local model, while structured options and relationship extraction use the API. These three modes can be toggled with one click via NLG_MODE in config.py.

---

## Slide 7: Evaluation & Quantitative Results

**Visual Suggestion:** Dashboard-style layout. Left: KG On/Off comparison bar chart (Consistency, Entity Coverage, Distinct-2, Self-BLEU). Right: LLM-as-Judge scoring dashboard (9.12/10).

**Slide Content:**
* **KG On vs Off Benchmarks:**
  * Consistency Rate: 0% → **100%** (+100%)
  * Entity Coverage: 0.79 → **1.00** (+26%)
  * Distinct-2: 0.64 → **0.74** (+15%)
  * Self-BLEU: 0.36 → **0.22** (-38%, lower is better)
* **LLM-as-Judge Score:** 7.50 → **9.12 / 10** (+21%)
* **Evaluation Metrics:** Distinct-N (lexical diversity), Self-BLEU (intra-session diversity), Entity Coverage (KG entity utilization), Consistency Rate (% of conflict-free turns)

**Speaker Notes (approx. 60 seconds):**
To ensure academic rigor, we implemented a comprehensive evaluation suite. Since interactive narratives lack a single "correct answer," we utilized reference-free metrics and comparative experiments.

We conducted critical KG On/Off benchmarks. The results show that enabling the Knowledge Graph skyrocketed narrative consistency from 0% to 100%—meaning logical contradictions were completely eliminated. Entity coverage increased by 26%, proving the generated story actively utilizes world details from the graph. Distinct-2 improved by 15% and Self-BLEU decreased by 38%, indicating the LLM no longer falls into repetitive loops.

Furthermore, we used the LLM-as-Judge method for human-like quality assessment, and the overall score rose from 7.50 to 9.12 out of 10. These figures strongly prove the Knowledge Graph's substantial contribution to generation quality.

---

## Slide 8: Conclusion & Future Work

**Visual Suggestion:** Clean summary page. Left lists core achievements, right lists future directions. Bottom features a GitHub QR code and "Thank You / Q&A".

**Slide Content:**
* **Core Achievements:**
  * Built a complete NLU-KG-NLG hybrid narrative engine
  * Knowledge Graph boosted consistency from 0% to 100%, lowered Self-BLEU by 38%
  * Dual-Channel Conflict Detection: Deterministic Rules + LLM Arbitration
  * Lazy loading + transparent fallbacks ensure 100% availability
  * Interactive PyVis KG visualization for real-time world state debugging
* **Future Directions:**
  * Local model fine-tuning (reducing API dependency)
  * Multiplayer collaborative storytelling
  * Finer-grained emotion-driven narrative mechanics
* **Q&A:** Thank you for listening!

**Speaker Notes (approx. 45 seconds):**
In conclusion, StoryWeaver successfully demonstrates that combining deterministic NLU models and a structured Knowledge Graph with generative LLMs can significantly enhance the coherence and quality of AI narratives. Our Knowledge Graph brought consistency from 0% up to 100%, the dual-channel conflict detection ensured logical harmony, and the lazy loading/fallback mechanisms guaranteed robust engineering.

In the future, we plan to further fine-tune local models to reduce API dependencies, explore multiplayer collaborative narrative branches, and introduce more granular emotion-driven storytelling mechanics.

Thank you all for your time and attention. We would now be happy to answer any questions.

---

## Appendix: Presentation Time Allocation

| Slide | Content | Duration |
|-------|---------|----------|
| 1 | Title Page | 30s |
| 2 | Motivation & Problem | 50s |
| 3 | System Architecture | 60s |
| 4 | NLU Pipeline | 55s |
| 5 | Knowledge Graph | 75s |
| 6 | NLG Generation | 55s |
| 7 | Evaluation Results | 60s |
| 8 | Conclusion & Q&A | 45s |
| **Total** | | **Approx. 7 min 10s** |

---

## Appendix: Six Highlight Coverage Checklist

| Highlight | Coverage Location | Status |
|-----------|-------------------|--------|
| KG On/Off Quantitative Data | Slide 5, 7, 8 | ✅ |
| Dual-Channel Conflict Detection | Slide 3, 5 | ✅ |
| Lazy Loading & Transparent Fallback | Slide 3, 4 | ✅ |
| Three NLG Modes | Slide 3, 6 | ✅ |
| Emotion Analysis (DistilRoBERTa, 7 Ekman) | Slide 3, 4, 6 | ✅ |
| Interactive PyVis KG Visualization | Slide 3, 5 | ✅ |