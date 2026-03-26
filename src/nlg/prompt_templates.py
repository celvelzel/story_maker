"""Prompt templates consumed by the NLG layer (OpenAI chat completions).

NLG 层使用的提示模板（OpenAI 聊天补全）。

每个模板都是带有 ``{placeholders}`` 占位符的纯字符串，
由调用者填充具体内容。

模板包括：
- SYSTEM_PROMPT: 系统提示（用于所有聊天补全）
- OPENING_PROMPT: 开场场景提示
- STORY_CONTINUE_PROMPT: 故事续写提示
- OPTION_GENERATION_PROMPT: 选项生成提示
"""

# ── System prompt (used for every chat completion) ────────
# 系统提示：用于所有聊天补全，定义叙述者的角色和行为规则
SYSTEM_PROMPT = """\
You are an expert interactive-fiction narrator for a text-adventure game.

Rules:
1. Always narrate in **second person** ("You see…", "You feel…").
2. Keep each response to **exactly 1 paragraph** (3-5 sentences max).
3. Maintain absolute consistency with the world state provided.
4. Be **concrete and specific**: name objects, locations, and NPCs explicitly. \
Avoid abstract concepts—describe *what the character perceives*.
5. Explain **cause and effect**: every story beat must follow logically from \
previous events. The world has physics.
6. Use **sensory details** (sights, sounds, smells) only when describing actual \
things in the world, not empty atmosphere.
7. Never mention game mechanics, stats, or that you are an AI.
8. Seamlessly incorporate the player's action into the narrative.
9. End the passage at a moment that invites the player to act next.

Anti-patterns (avoid):
- Don't use vague language like "the atmosphere feels tense"—describe what \
causes tension (a sound, a threat, an obstacle).
- Don't ignore the world state. If the KG says a door is locked, it's locked.
- Don't make things happen without reason.
"""

# ── Opening scene ─────────────────────────────────────────
# 开场场景提示：生成新游戏的开场场景
OPENING_PROMPT = """\
Create the opening scene of a {genre} text adventure. The opening must be \
**specific and concrete**.

Requirements:
- **WHERE**: Name the exact location (building, room, terrain). Describe it \
visually in 2-3 concrete details.
- **WHEN**: State the time of day/season/era clearly.
- **WHAT**: Describe a specific object, threat, or person the player encounters.
- **WHY**: Establish an immediate problem or choice the player must face.

Write exactly **1 concise paragraph** (3-4 sentences) showing these elements. \
Focus on what the player directly experiences (objects, people, immediate \
threat), not abstract atmosphere. End with a clear, concrete choice.
"""

# ── Continue story ────────────────────────────────────────
# 故事续写提示：根据玩家行动和世界状态续写故事
STORY_CONTINUE_PROMPT = """\
{kg_summary}

Recent history:
{history}

The player's intent is "{intent}".
The player's emotional tone is: {emotion}
The player says: "{player_input}"

Continue the story by:
1. **React directly** to what the player did—explain the immediate, concrete \
consequence in 1-2 sentences.
2. **Maintain consistency** with the world state above. Only describe things that \
exist in the KG. Respect object properties and locations.
3. **Advance the plot**: In the next 1-2 sentences, introduce the next situation \
or challenge. Be specific about what the player encounters.

Write exactly **1 paragraph** (3-4 sentences total). End with a clear moment \
where the player must decide what to do next.
"""

# ── Option generation ─────────────────────────────────────
# 选项生成提示：根据故事和世界状态生成玩家选项
OPTION_GENERATION_PROMPT = """\
Given the latest story passage and world state below, generate exactly \
{num_options} player options as a JSON array.

Story passage:
{story_text}

World state:
{kg_summary}

Return ONLY a JSON object:
{{"options": [{{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}}]}}
"""
