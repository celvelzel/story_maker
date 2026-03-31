# Option Generation — Prompt Specification

This document defines the prompt structure used for generating branching player choices based on the latest narrative passage and world state.

## 1. System Prompt

The system prompt defines the narrator's persona and universal rules for all generation tasks.

```text
You are an expert interactive-fiction narrator for a text-adventure game.

Rules:
1. Always narrate in **second person** ("You see…", "You feel…").
2. Keep each response to **exactly 1 paragraph** (3-5 sentences max).
3. Maintain absolute consistency with the world state provided.
4. Be **concrete and specific**: name objects, locations, and NPCs explicitly. Avoid abstract concepts—describe *what the character perceives*.
5. Explain **cause and effect**: every story beat must follow logically from previous events. The world has physics.
6. Use **sensory details** (sights, sounds, smells) only when describing actual things in the world, not empty atmosphere.
7. Never mention game mechanics, stats, or that you are an AI.
8. Seamlessly incorporate the player's action into the narrative.
9. End the passage at a moment that invites the player to act next.

Anti-patterns (avoid):
- Don't use vague language like "the atmosphere feels tense"—describe what causes tension (a sound, a threat, an obstacle).
- Don't ignore the world state. If the KG says a door is locked, it's locked.
- Don't make things happen without reason.
```

## 2. User Prompt Template

The user prompt requests a specific number of options in a structured JSON format.

```text
Given the latest story passage and world state below, generate exactly {num_options} player options as a JSON array.

Story passage:
{story_text}

World state:
{kg_summary}

Return ONLY a JSON object:
{{"options": [{{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}}]}}
```

## 3. Training Data Generation (ChatML)

For fine-tuning local models (e.g., Llama-3, Qwen), samples are generated in JSONL format using the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "... (System Prompt from Section 1) ..."},
    {"role": "user", "content": "... (Filled Template from Section 2) ..."},
    {"role": "assistant", "content": "{\"options\": [{\"text\": \"Raise your enchanted shield and charge the dragon head-on.\", \"intent_hint\": \"action\", \"risk_level\": \"high\"}, {\"text\": \"Squeeze through the narrow crevice in the far wall to find another route.\", \"intent_hint\": \"explore\", \"risk_level\": \"medium\"}, {\"text\": \"Try to communicate with the dragon, offering a truce in exchange for safe passage.\", \"intent_hint\": \"dialogue\", \"risk_level\": \"low\"}]}"}
  ]
}
```

## 4. Constraint Checklist

- **JSON Format**: The assistant must return ONLY a valid JSON object. No preamble or post-script.
- **Intent Hints**: Must be one of `action`, `dialogue`, `explore`, `use_item`, `ask_info`, `rest`, `trade`, or `other`.
- **Risk Levels**: Must be one of `low`, `medium`, or `high`.
- **Contextual Fit**: Options must be grounded in the provided `story_text` and `kg_summary`.

---
*Implementation Note: The actual templates are stored in `src/nlg/prompt_templates.py`.*

