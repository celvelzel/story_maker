# StoryWeaver: AI-Powered Text Adventure Game
## Presentation Script & Speaker Notes

**Target Duration:** ~7 Minutes
**Total Word Count:** ~950 words

---

### Slide 1: Title Slide
**Slide Title:** StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation
**Visual Suggestion:** A cinematic, split-screen title slide. On the left, a retro-style text adventure interface with glowing green text. On the right, a futuristic visualization of a neural network or a glowing node-based knowledge graph. Include the project name "StoryWeaver," the course code "COMP5423," and the team members' names.
**Slide Content:**
* StoryWeaver: AI-Powered Text Adventure Game
* Dynamic Plot Generation & Interactive Storytelling
* COMP5423 NLP Group Project
* [Team Members' Names]

**Speaker Notes:** (Estimated time: 30 seconds | ~60 words)
"Good evening, everyone. Welcome to our presentation for the COMP5423 NLP Group Project. Today, our team is thrilled to introduce 'StoryWeaver'—an AI-powered text adventure game engine that we have built from the ground up. Our goal with this project is to redefine interactive storytelling by using advanced Natural Language Processing techniques to create dynamic, responsive, and infinitely branching narratives."

---

### Slide 2: Introduction & Problem Statement
**Slide Title:** The Evolution of Interactive Fiction
**Visual Suggestion:** A comparative diagram. On one side, "Traditional Games" showing a rigid, pre-written flowchart of static story branches. On the other side, "StoryWeaver" showing an open-ended, expanding web of dynamic, AI-generated possibilities.
**Slide Content:**
* **Traditional Text Adventures:** Rigid, pre-scripted branches, limited player agency.
* **The Challenge:** Maintaining narrative consistency while allowing infinite player freedom.
* **Our Solution:** StoryWeaver—combining local NLU models with LLM generation for infinite, coherent storytelling.

**Speaker Notes:** (Estimated time: 45 seconds | ~100 words)
"Let's start by looking at the problem. Traditional text adventure games rely on pre-written scripts. The player’s choices are limited to a rigid, hardcoded flowchart. If a player tries something the developer didn't anticipate, the game simply fails to understand. The core challenge in modernizing this genre is how to grant the player complete freedom of input without breaking the game's narrative consistency. Our solution is StoryWeaver. By integrating local Natural Language Understanding models with Large Language Models, we shift the paradigm from 'pre-scripted branching' to 'dynamic plot generation,' allowing for truly personalized and coherent storytelling based entirely on what the player types."

---

### Slide 3: Project Requirements & Core Objectives
**Slide Title:** Fulfilling the COMP5423 Requirements
**Visual Suggestion:** A clean, four-pillar diagram. Each pillar represents a core requirement: "Intent Recognition," "Context-Aware Generation," "Plot Consistency," and "Dialogue Management," with small, intuitive icons for each.
**Slide Content:**
* **Understand Player Input:** Intent recognition and entity extraction.
* **Dynamic Generation:** Context-aware story continuation using LLMs.
* **Narrative Consistency:** Dynamic Knowledge Graph to prevent plot holes.
* **Evaluation & UX:** Measuring coherence, responsiveness, and player immersion.

**Speaker Notes:** (Estimated time: 45 seconds | ~100 words)
"To achieve this, we aligned our development strictly with the COMP5423 project specifications. Our system needed to seamlessly execute four core NLP tasks. First, understanding user intent—interpreting exactly what the player wants to do. Second, context-aware text generation—crafting story responses that fit the current situation. Third, and perhaps most importantly, plot consistency—ensuring the AI remembers past events so characters don't magically come back to life. Finally, we needed a robust dialogue management system to tie the interactions together. We built our architecture to hit every single one of these marks efficiently."

---

### Slide 4: Technical Architecture Overview
**Slide Title:** The Engine Behind StoryWeaver
**Visual Suggestion:** A high-level system architecture flowchart based on the README. showing the Streamlit Frontend connecting to the Game Engine orchestrator, which then branches to the local NLU module (DistilBERT/spaCy), the NLG module (OpenAI/Local LLM), and the Knowledge Graph state manager.
**Slide Content:**
* **Frontend:** Interactive Streamlit UI with Knowledge Graph Visualizer.
* **NLU Pipeline (Local):** DistilBERT (Intent), spaCy (NER), fastcoref (Coreference).
* **NLG Pipeline (API/Local):** OpenAI API / Qwen for story & option generation.
* **State Management:** NetworkX Knowledge Graph for conflict detection.

**Speaker Notes:** (Estimated time: 60 seconds | ~130 words)
"This is the architecture powering StoryWeaver. At the top, we have our Streamlit frontend, which provides a clean chat interface and real-time debug visualizations. When a player inputs a command, it hits our Game Engine Orchestrator. We designed a hybrid approach for optimal performance. The Natural Language Understanding pipeline runs locally—using a fine-tuned DistilBERT for intent classification, spaCy for entity recognition, and fastcoref to resolve pronouns like 'it' or 'him'. Once we understand the input, we pass the structured data to our Natural Language Generation module, powered by advanced LLMs, to continue the story and generate new choices. All of this is anchored by a dynamic Knowledge Graph that constantly updates the world state."

---

### Slide 5: The Per-Turn Processing Pipeline
**Slide Title:** Step-by-Step: How a Turn Unfolds
**Visual Suggestion:** A horizontal timeline or step-by-step pipeline graphic (Steps 1 to 5). Highlight an example: Player types "Hit the goblin with my sword." -> NLU extracts intent (Attack) -> State checks Graph -> LLM generates outcome -> Graph updates goblin to "defeated."
**Slide Content:**
1. **Coreference Resolution:** Resolving pronouns using recent history.
2. **Intent & Entity Extraction:** Parsing actions and targets.
3. **Knowledge Graph Check & Update:** Preventing logical conflicts.
4. **Story Generation:** Crafting the narrative response.
5. **Option Generation:** Providing 3 dynamic, risk-assessed choices.

**Speaker Notes:** (Estimated time: 60 seconds | ~140 words)
"Let’s walk through what happens under the hood during a single turn. Imagine the player types, 'Take the key and unlock it.' Step one is Coreference Resolution; our fastcoref model determines that 'it' refers to the 'rusty door' from the previous turn. Step two is Intent and Entity Extraction, where DistilBERT identifies the intent as 'unlock' and the entities as 'key' and 'door'. In step three, before we write the story, we query our Knowledge Graph to ensure the player actually has the key. If there are no conflicts, step four triggers the LLM to write a contextual, engaging narrative response. Finally, in step five, the system generates three new possible actions for the player, categorized by risk level, to keep the gameplay moving."

---

### Slide 6: Dynamic Knowledge Graph (The Memory)
**Slide Title:** Maintaining Narrative Consistency
**Visual Suggestion:** A zoomed-in, interactive-looking network node graph. Show nodes like "Player," "Village Elder," and "Forbidden Cave," connected by edges like "Has spoken to," "Possesses," or "Located in."
**Slide Content:**
* **The Problem with LLMs:** Prone to hallucination and forgetting context.
* **Our Solution:** A NetworkX-powered explicit world state.
* **Mechanism:** Relation extraction from text to graph nodes/edges.
* **Result:** Rules-based and LLM-assisted conflict detection prevents impossible actions.

**Speaker Notes:** (Estimated time: 60 seconds | ~140 words)
"A major challenge in AI storytelling is that LLMs hallucinate or forget long-term context. To solve this, we implemented a Dynamic Knowledge Graph using NetworkX. Instead of relying purely on the LLM's context window, our system extracts entities and relationships from every generated story segment and maps them explicitly. If the player acquires an item, it becomes a node connected to the player. When a player attempts an action, our engine cross-references the Knowledge Graph first. If the player tries to 'negotiate with the village elder' but the graph shows the elder is in a different location, the system detects the conflict and forces the story to reflect that failure, ensuring a logically sound game world."

---

### Slide 7: Evaluation & Performance
**Slide Title:** System Evaluation & Metrics
**Visual Suggestion:** A dashboard-style layout showing bar charts or gauges for NLU Accuracy, Response Latency, and Text Diversity (Distinct-n/Self-BLEU scores), alongside a "User Satisfaction" metric.
**Slide Content:**
* **NLU Accuracy:** High precision on intent classification via fine-tuned DistilBERT.
* **Text Diversity:** Evaluated using Distinct-n and Self-BLEU metrics.
* **Quality Assurance:** LLM-as-a-Judge to measure narrative coherence.
* **Performance:** Optimized local NLP execution for low latency.

**Speaker Notes:** (Estimated time: 45 seconds | ~110 words)
"To ensure we built a robust system and not just a prototype, we implemented rigorous evaluation metrics. We evaluate our text generation using Distinct-n and Self-BLEU scores to guarantee lexical diversity—making sure the AI doesn't sound repetitive. We also utilize an 'LLM-as-a-Judge' approach to automatically score the plot coherence and logical flow of the generated stories. On the performance side, moving our NLU components to local, lightweight models like DistilBERT significantly reduced our pipeline latency, resulting in a snappy, responsive user experience that keeps the player immersed."

---

### Slide 8: Challenges & Solutions
**Slide Title:** Overcoming Development Hurdles
**Visual Suggestion:** A split slide. Left side: "Challenges" with icons of bugs, slow clocks, and tangled wires. Right side: "Solutions" with icons of speedometers, organized folders, and checkmarks.
**Slide Content:**
* **Challenge:** LLM context window limits and high latency.
  * *Solution:* Shifted NLU locally; explicit Knowledge Graph memory.
* **Challenge:** Pronoun ambiguity in player input.
  * *Solution:* Integrated fastcoref for highly accurate contextual resolution.
* **Challenge:** Unpredictable player inputs breaking the game state.
  * *Solution:* Multi-layered conflict detection pipeline.

**Speaker Notes:** (Estimated time: 50 seconds | ~110 words)
"Building StoryWeaver came with significant challenges. Initially, sending everything to an LLM was too slow and expensive. We solved this by decoupling the NLU pipeline and running models locally, which drastically cut down latency. Another major hurdle was pronoun ambiguity. Players naturally type things like 'attack him', but the engine needs to know who 'him' is. Integrating the fastcoref library completely solved this by tracking entities across the conversation history. Finally, players are unpredictable. To handle wild inputs, our multi-layered conflict detection ensures that even if a player tries to do something impossible, the game engine catches it gracefully and incorporates the failure into the story."

---

### Slide 9: Conclusion & Future Work
**Slide Title:** The Future of StoryWeaver
**Visual Suggestion:** A forward-looking, inspiring graphic—perhaps a player looking into a glowing book or a portal. A QR code to the project's GitHub repository in the corner.
**Slide Content:**
* **Summary:** Successfully bridged theoretical NLP with an interactive application.
* **Impact:** A scalable framework for dynamic storytelling and training datasets.
* **Future Work:** Multi-agent NPC interactions, local LLM fine-tuning, and voice integration.
* **Thank You! / Q&A**

**Speaker Notes:** (Estimated time: 45 seconds | ~100 words)
"In conclusion, StoryWeaver successfully bridges theoretical NLP foundations with a highly interactive, real-world application. We delivered a system that understands intent, generates compelling narratives, and maintains strict logical consistency. Looking ahead, this architecture lays the groundwork for exciting future expansions, such as integrating multi-agent AI for complex NPC behaviors, fully local LLM execution, or even voice-to-text integration. We have learned an immense amount about system architecture, NLP integration, and teamwork throughout this project. Thank you very much for your time and attention. We would now be happy to show you a live demonstration and answer any questions."
