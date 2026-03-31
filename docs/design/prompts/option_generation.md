# Option Generation — Prompt Specification

> **Last Updated:** 2026-04-01  
> **Source:** `src/nlg/prompt_templates.py`

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
{"options": [{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}]}
```

## 3. Output Schema

Each option is a `StoryOption` dataclass:

```python
@dataclass
class StoryOption:
    text: str            # Option text displayed to user
    intent_hint: str     # Suggested intent category
    risk_level: str      # Risk level: "low" | "medium" | "high"
```

## 4. Constraint Checklist

- **JSON Format**: The assistant must return ONLY a valid JSON object. No preamble or post-script.
- **Intent Hints**: Must be one of `action`, `dialogue`, `explore`, `use_item`, `ask_info`, `rest`, `trade`, or `other`.
- **Risk Levels**: Must be one of `low`, `medium`, or `high`.
- **Contextual Fit**: Options must be grounded in the provided `story_text` and `kg_summary`.

## 5. Fallback Behavior

If LLM option generation fails, the system falls back to hardcoded defaults:

```python
_FALLBACK_OPTIONS = [
    StoryOption("Look around and assess the situation.", "explore", "low"),
    StoryOption("Move cautiously forward.", "action", "medium"),
    StoryOption("Try to speak with someone nearby.", "dialogue", "low"),
]
```

## 6. Usage

```python
from src.nlg.option_generator import OptionGenerator
from src.nlg.prompt_templates import OPTION_GENERATION_PROMPT, SYSTEM_PROMPT

option_gen = OptionGenerator()
options = option_gen.generate(story_text, kg_summary, num_options=3)
```

---
*Implementation Note: The actual templates are stored in `src/nlg/prompt_templates.py`.*

