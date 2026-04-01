# StoryWeaver: Presentation Script

## Slide 1: Title & Introduction
**Visual Suggestion:** A sleek, dark-themed title slide with a subtle fantasy or sci-fi text adventure aesthetic. The game title "StoryWeaver" should be prominent, with a dynamic, glowing "AI" node network graphic subtly connecting the letters. Include university/course branding (COMP5423) and group member names at the bottom.

**Slide Content:**
* **StoryWeaver**
* AI-Powered Text Adventure Game with Dynamic Plot Generation
* COMP5423 NLP Group Project
* Group Members: [List your group members from the intro doc]

**Speaker Notes:**
*(Estimated time: 45 seconds)*
"Good evening, everyone. Welcome to our presentation for COMP5423. Today, our team is thrilled to introduce 'StoryWeaver'—an AI-powered text adventure game engine that utilizes Natural Language Processing to generate dynamic, responsive plots based entirely on player choices. My name is [Your Name], and along with my teammates, we will walk you through the motivation, architecture, and technical implementation of a system that bridges theoretical NLP foundations with real-world interactive gaming. Let's dive into how we are redefining narrative gameplay."

---

## Slide 2: Motivation & Problem Statement
**Visual Suggestion:** A split-screen comparison graphic. On the left, a traditional "decision tree" showing a rigid, finite number of pre-written game endings (labeled "Static Narrative"). On the right, a glowing, infinitely expanding neural network or branching path (labeled "Dynamic Plot Generation"). 

**Slide Content:**
* **The Challenge:** Traditional text adventures rely on static, hard-coded branching paths.
* **Our Solution (StoryWeaver):** Infinite, personalized narrative generation using NLP.
* **Core Objectives:**
  * Understand complex user intents.
  * Maintain strict narrative consistency.
  * Produce contextually coherent story branches.

**Speaker Notes:**
*(Estimated time: 60 seconds)*
"Traditionally, text adventure games have relied on massive, hard-coded decision trees. While effective, they are limited by the developer's time and imagination. Our motivation with StoryWeaver was to break this limitation. By leveraging state-of-the-art Natural Language Processing, we've designed a system that acts as a real-time Dungeon Master. It doesn't just pick from a list of pre-written outcomes; it understands the player's natural language input, maintains narrative consistency over long sessions, and generates entirely new story segments that logically align with the game's setting. This project allows us to consolidate our NLP expertise into a highly engaging, immersive application."

---

## Slide 3: Project Requirements & Key Steps
**Visual Suggestion:** A four-pillar infographic or a timeline chevron diagram detailing the four major phases: Data Preparation, Algorithm Design, System Implementation, and Performance Evaluation. Use clean icons for each (e.g., a database icon, a brain/gear icon, a code window icon, and a bar chart icon).

**Slide Content:**
* **Data Preparation:** Text adventure scripts, dialogue datasets, and plot annotations.
* **Algorithm Design:** Context-aware generation, intent recognition, and dialogue management.
* **System Implementation:** Orchestrated via HuggingFace, PyTorch, and a Streamlit UI.
* **Performance Evaluation:** Assessing narrative quality, interaction responsiveness, and coherence.

**Speaker Notes:**
*(Estimated time: 60 seconds)*
"To successfully build StoryWeaver, our development process was strictly aligned with the course's task specifications, broken down into four key pillars. First, Data Preparation: organizing text adventure scripts and annotating plot consistency. Second, Algorithm Design: integrating local context-aware text generation and user intent recognition. Third, System Implementation: bringing the algorithms to life using frameworks like PyTorch and HuggingFace Transformers, all wrapped in a user-friendly Streamlit frontend. Finally, Performance Evaluation, where we rigorously assess the system’s plot coherence, response times, and overall player immersion."

---

## Slide 4: Technical Architecture Overview
**Visual Suggestion:** A clean, high-contrast block diagram of the system architecture (matching the README). The top layer shows the "Streamlit Frontend" (Chat UI, NLU Debug, Knowledge Graph). An arrow points down to the "Game Engine (Orchestrator)", which houses the local NLU modules (DistilBERT, spaCy, fastcoref) and the NLG modules (LLM API). A separate block shows the Knowledge Graph.

**Slide Content:**
* **Streamlit Frontend:** Interactive Chat UI, NLU Debugger, and live Knowledge Graph visualizer.
* **Game Engine (Orchestrator):** The brain coordinating state, input, and generation.
* **Local NLU Module:** DistilBERT + spaCy + fastcoref for fast, cost-effective intent & entity extraction.
* **NLG & Knowledge Graph:** LLM-powered story APIs integrated with a NetworkX dynamic world state.

**Speaker Notes:**
*(Estimated time: 90 seconds)*
"Here is the high-level architecture of StoryWeaver. We designed the system to be highly modular and efficient. At the top, we have our Streamlit Frontend, providing not just the game interface, but live debugging tools for the NLU pipeline and a visualizer for our Knowledge Graph. 
The core is our Game Engine Orchestrator. To balance performance and capability, we utilize a hybrid approach. For Natural Language Understanding—parsing what the player actually means—we use fast, local models including a fine-tuned DistilBERT for intent classification, spaCy for entity extraction, and fastcoref for coreference resolution. For the heavy lifting of creative writing, our Natural Language Generation module utilizes powerful external LLM APIs, tightly constrained by our internal game state."

---

## Slide 5: The Core Pipeline (Per Turn)
**Visual Suggestion:** A sleek horizontal flowchart showing a single player turn. Player Input -> Coreference Resolution -> Intent & Entity Extraction -> Knowledge Graph Check -> Story Generation -> Option Generation -> UI Update. Highlight the transition from NLU (understanding) to NLG (creating).

**Slide Content:**
* **1. Resolution & Extraction:** Resolving pronouns (fastcoref) and identifying intents/entities (DistilBERT/spaCy).
* **2. Knowledge Graph Update:** Extracting new relations and detecting plot conflicts.
* **3. Story Generation:** LLM continues the narrative based strictly on current constraints.
* **4. Option Generation:** System dynamically generates three logical choices with varying risk levels.

**Speaker Notes:**
*(Estimated time: 90 seconds)*
"Let’s look at what happens under the hood during a single turn of gameplay. When a player inputs an action, the pipeline immediately runs Coreference Resolution to understand pronouns based on recent history. Next, DistilBERT and spaCy classify the intent and extract entities. 
Before any text is generated, our system consults the Knowledge Graph. It checks for narrative conflicts—ensuring you can't talk to an NPC that is currently in another location. Once the logic is verified, the LLM generates the next story segment. Finally, the Option Generation module provides three dynamic, risk-varying choices for the player, ensuring the gameplay loop remains continuous and engaging."

---

## Slide 6: Managing Narrative Consistency (Knowledge Graph)
**Visual Suggestion:** A stylized representation of a Knowledge Graph (nodes connected by edges). Show a specific example: A player node connected to an item node ("Player -> holds -> Magic Sword") and an environment node ("Goblin -> hates -> Magic Sword"). 

**Slide Content:**
* **The "Hallucination" Problem:** Unconstrained LLMs often forget plot points or contradict themselves.
* **Dynamic World State:** NetworkX manages entities, locations, and relationships.
* **Conflict Detection:** Rule-based logic paired with LLM consistency checking.
* **Result:** A persistent, logical game world that remembers player actions.

**Speaker Notes:**
*(Estimated time: 60 seconds)*
"One of the biggest challenges in AI narrative generation is the 'hallucination' problem—where the AI forgets past events or contradicts the established plot. We solved this by implementing a Dynamic Knowledge Graph using NetworkX. 
Instead of just feeding the LLM raw text history, our pipeline extracts entities and relationships from every turn and updates a structured database. If a player tries to use an item they don't possess, our Conflict Detection module catches it before the story generator even runs. This ensures that the generated world is persistent, logical, and deeply reactive to the player's specific journey."

---

## Slide 7: Conclusion & Live Demonstration
**Visual Suggestion:** A bold summary slide with a "Play" or "Start Game" graphic. A QR code or link to the GitHub repository can be placed in the corner. Key takeaways listed clearly.

**Slide Content:**
* **Bridging Theory & Practice:** Combining robust local NLU with generative LLMs.
* **Scalable Architecture:** Modular design allows for easy swapping of models.
* **Future Work:** Expanding the evaluation metrics and multi-agent NPC interactions.
* **Live Demonstration:** Let's see StoryWeaver in action!

**Speaker Notes:**
*(Estimated time: 45 seconds)*
"In conclusion, StoryWeaver successfully bridges theoretical NLP concepts with an engaging, interactive application. By combining lightweight, local NLU models with the creative power of generative LLMs, we've created a scalable architecture that maintains strict narrative consistency without sacrificing creativity. 
We look forward to expanding our evaluation metrics and potentially adding multi-agent NPC interactions in the future. We believe this project perfectly encapsulates the intended learning outcomes of this course. Thank you for your time and attention. Now, we would like to transition to a live demonstration to show you StoryWeaver in action."
