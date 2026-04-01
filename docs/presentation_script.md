# StoryWeaver Presentation Script

This document contains the presentation script and speaker notes for the COMP5423 NLP Group Project.
Target Time: 7 minutes (~900 - 1,000 words).

## Slide 1: Title Slide
**Visual Suggestion:** A sleek, cyberpunk-themed title slide with the game's logo "StoryWeaver". Include a subtle background image of a glowing, interconnected knowledge graph or a digital book. Include group members' names and student IDs.

**Slide Content:**
* StoryWeaver: AI-Powered Text Adventure Game
* Dynamic Plot Generation with NLU & Knowledge Graphs
* COMP5423 NLP Group Project
* Group [Your Group ID]

**Speaker Notes (approx. 30 seconds / 65 words):**
Good evening, everyone. We are Group [Your Group ID], and today we are thrilled to introduce StoryWeaver. StoryWeaver is an AI-powered text adventure game engine that we developed for our COMP5423 NLP project. Our goal was to push the boundaries of interactive storytelling by combining local Natural Language Understanding models with Large Language Models and dynamic Knowledge Graphs to create a truly personalized and consistent gaming experience.

---

## Slide 2: Motivation & Problem Statement
**Visual Suggestion:** A split-screen comparison. Left side: traditional text games (rigid decision trees, "wall of text", limited options). Right side: StoryWeaver (dynamic branching, AI brain, infinite possibilities).

**Slide Content:**
* **Traditional Text Adventures:**
  * Rigid decision trees
  * Limited, pre-written choices
  * Memory loss over long sessions
* **The StoryWeaver Solution:**
  * Infinite, dynamically generated plotlines
  * Natural language player input understanding
  * Persistent world state and narrative consistency

**Speaker Notes (approx. 45 seconds / 100 words):**
The motivation behind this project comes from the limitations of traditional text adventure games. Historically, these games rely on rigid, hard-coded decision trees. If a player wants to do something outside the pre-written script, the game simply says "I don't understand." Furthermore, as stories get longer, basic AI generators often suffer from "hallucinations" or forget earlier plot points. 

With StoryWeaver, we solve these problems. We built a system that understands natural language input, generates infinite dynamic plotlines based on your actions, and most importantly, maintains strict narrative consistency using a persistent world state. You aren't just selecting options; you are co-authoring the story.

---

## Slide 3: Technical Architecture
**Visual Suggestion:** A clean, high-level architecture diagram showing the data flow. Player Input -> NLU Module (DistilBERT/spaCy/fastcoref) -> Game Engine -> Story/Option Gen (LLM) -> Knowledge Graph. Show Streamlit wrapping everything.

**Slide Content:**
* **Frontend:** Streamlit Cyberpunk UI (Chat, KG Visualizer, Debugger)
* **Orchestrator:** Game Engine coordinating the per-turn pipeline
* **NLU (Local):** DistilBERT (Intent), spaCy (NER), fastcoref (Coreference)
* **NLG (Cloud/Local):** OpenAI GPT-4o-mini / Local Qwen
* **World State:** NetworkX Knowledge Graph & Conflict Detector

**Speaker Notes (approx. 60 seconds / 130 words):**
Here is the high-level architecture of StoryWeaver. At the core is our Game Engine orchestrator. When a player submits a turn via our Streamlit frontend, the input flows through three main phases. 

First, the local Natural Language Understanding, or NLU pipeline, processes the text to understand *what* the player wants to do. 
Second, this structured data, along with the current world state, is sent to our Natural Language Generation module—powered by an LLM—to generate the next story segment and three dynamic choices.
Finally, the new story events are parsed and injected into our dynamic Knowledge Graph, which tracks entities and relationships to ensure the world remains consistent. This modular design allows us to swap components, like using a local Qwen model instead of the OpenAI API.

---

## Slide 4: NLU Pipeline - Understanding the Player
**Visual Suggestion:** A flowchart showing an example sentence (e.g., "I attack the goblin with it") transforming as it passes through Coreference, Intent Classification, and Entity Extraction. Include a small code snippet showing the pipeline sequence.

**Slide Content:**
* **Coreference Resolution (fastcoref):** "I attack it" -> "I attack the goblin"
* **Intent Classification (DistilBERT):** Maps input to 8 distinct actions (e.g., *attack*, *explore*, *dialogue*)
* **Entity Extraction (spaCy):** Identifies custom game entities (Person, Location, Item) with fallback heuristics

**Speaker Notes (approx. 60 seconds / 140 words):**
To truly understand player input, our NLU pipeline performs three critical tasks sequentially. 

First, we tackle pronoun ambiguity using the `fastcoref` neural model. If a player says "I attack it," the system looks at the recent story context and resolves "it" to "the goblin."
Next, we pass this resolved text to a DistilBERT model that we fine-tuned specifically for this project. It classifies the player's intent into one of eight categories, such as combat, exploration, or dialogue, with high confidence.
Finally, we use `spaCy`'s Named Entity Recognition, enhanced with custom noun-phrase heuristics, to extract specific game entities like items, locations, and characters. By combining these three local NLP models, the engine precisely understands the player's action before passing it to the LLM for story generation.

---

## Slide 5: Maintaining Consistency with the Knowledge Graph
**Visual Suggestion:** A visual representation of a graph network (nodes and edges) with properties. Show how an entity like "Village Elder" has properties like status, location, and relationships. Highlight a "Conflict Detected!" warning box.

**Slide Content:**
* **NetworkX MultiDiGraph:** Tracks characters, items, and locations over time
* **Temporal Decay & Importance Scoring:** Prioritizes recent and highly-mentioned entities
* **Conflict Detection:** Identifies contradictions (e.g., a dead character speaking)
* **Layered Summaries:** Feeds only relevant context to the LLM to prevent context-window bloat

**Speaker Notes (approx. 70 seconds / 150 words):**
One of the biggest challenges in AI story generation is consistency. To solve this, we implemented a dynamic Knowledge Graph using NetworkX. 

Instead of feeding the entire chat history to the LLM—which wastes tokens and confuses the model—we extract entities and relationships after every turn and update the graph. The graph features an importance scoring system that factors in temporal decay, meaning older, less relevant details fade into the background, while frequently mentioned entities stay prominent.

We also built a Conflict Detector. If the LLM tries to generate a story where a character who was previously killed suddenly starts talking, the conflict detector catches this contradiction against the graph's state and forces a correction. Finally, the graph generates a layered summary of the world state, providing the LLM with a strict, factual context for the next generation.

---

## Slide 6: NLG & Dynamic Story Generation
**Visual Suggestion:** A clean UI mockup or screenshot of the Streamlit interface, highlighting the generated story text and the three dynamically generated options with their risk levels.

**Slide Content:**
* **Context-Aware Generation:** Fuses player intent, KG summary, and emotion
* **Dynamic Options:** Generates 3 choices per turn with varying risk levels (Low/Medium/High)
* **LLM Agnostic:** Supports OpenAI GPT-4o-mini and local LLaMA/Qwen via llama.cpp
* **Interactive UI:** Streamlit interface with real-time KG visualization and NLU debugging

**Speaker Notes (approx. 60 seconds / 130 words):**
For the actual story generation, we engineered highly specific prompts that fuse the player's classified intent, their emotional tone, and the Knowledge Graph summary. This ensures the LLM generates a narrative continuation that strictly respects the established world rules.

Additionally, our Option Generator creates three distinct choices for the player at the end of each turn, categorized by risk level: low, medium, and high. This keeps the gameplay engaging even if the player doesn't want to type a custom response.

All of this is wrapped in a responsive Streamlit interface. The UI not only renders the chat but also provides a real-time, interactive visualization of the Knowledge Graph and NLU debug data, allowing us to see exactly how the AI is interpreting the game state under the hood.

---

## Slide 7: Evaluation & Quality Assessment
**Visual Suggestion:** A dashboard-style layout showing key metrics: Distinct-N charts, Self-BLEU scores, and a dial/gauge for Consistency Rate.

**Slide Content:**
* **Automated Reference-Free Metrics:**
  * **Distinct-N & Self-BLEU:** Measures vocabulary diversity and prevents repetitive loops
  * **Entity Coverage:** Ensures generated text utilizes KG entities
* **Consistency Tracking:** Measures the ratio of conflict-free turns
* **LLM-as-a-Judge:** Automated quality assessment using GPT-4 for human-like evaluation

**Speaker Notes (approx. 50 seconds / 115 words):**
To ensure our system meets high academic standards, we implemented a robust, automated evaluation suite. Because interactive stories don't have a single "correct" reference text, we rely on reference-free metrics.

We use Distinct-N and Self-BLEU to measure vocabulary diversity, ensuring the LLM doesn't fall into repetitive text loops. We also track Entity Coverage to verify that the generated story is actually utilizing the characters and items stored in our Knowledge Graph. 

Furthermore, we measure the system's consistency rate over long sessions, and utilize an LLM-as-a-Judge approach. This allows us to automatically evaluate narrative quality, coherence, and responsiveness to player choices in real-time, without requiring extensive manual human annotation.

---

## Slide 8: Conclusion & Future Work
**Visual Suggestion:** A forward-looking image (e.g., a glowing horizon or a character looking at a vast digital landscape). Simple, powerful bullet points summarizing the achievement.

**Slide Content:**
* **Achievement:** Built a fully integrated, context-aware NLP game engine
* **Key Takeaway:** Knowledge Graphs effectively ground LLMs, reducing hallucinations
* **Future Work:** 
  * Expand local model finetuning
  * Support multiplayer/co-op narratives
* **Q&A:** Thank you for playing!

**Speaker Notes (approx. 45 seconds / 100 words):**
In conclusion, StoryWeaver successfully demonstrates how combining traditional, deterministic NLP models like DistilBERT and spaCy with modern Generative LLMs creates a robust, interactive system. By implementing a dynamic Knowledge Graph, we effectively grounded the LLM, significantly reducing hallucinations and maintaining narrative consistency over long game sessions.

For future work, we plan to further fine-tune local models specifically for text adventure generation to reduce API dependency, and potentially explore multiplayer narrative branching.

Thank you for your time and attention. We hope you enjoyed this look into StoryWeaver, and we would now like to open the floor to any questions.
