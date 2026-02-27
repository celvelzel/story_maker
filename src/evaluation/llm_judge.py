"""LLM-as-Judge evaluation: rate a full story session on 5 dimensions.

Dimensions (each 1-10):
- narrative_quality
- consistency
- player_agency
- creativity
- pacing
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are an expert literary critic evaluating an interactive fiction session. "
    "Score each dimension from 1 (terrible) to 10 (masterful). "
    "Return ONLY valid JSON with the five keys."
)

_JUDGE_USER = """\
Below is the full transcript of an interactive story session.

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Rate the session on these five dimensions (1-10):
1. narrative_quality – prose quality, vivid descriptions, engaging language
2. consistency – characters, locations, and facts remain coherent
3. player_agency – how meaningfully the player's choices affected the story
4. creativity – originality of plot, settings, and characters
5. pacing – appropriate story momentum and tension management

Return JSON:
{{"narrative_quality": <int>, "consistency": <int>, "player_agency": <int>, "creativity": <int>, "pacing": <int>}}
"""

DIMENSIONS = [
    "narrative_quality",
    "consistency",
    "player_agency",
    "creativity",
    "pacing",
]


def judge(transcript: str) -> Dict[str, int | float]:
    """Send the transcript to the LLM and return dimension scores.

    Returns a dict with each dimension key (int 1-10) plus ``"average"`` (float).
    On failure returns all dimensions as 0.
    """
    from src.utils.api_client import llm_client

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER.format(transcript=transcript[:6000])},
    ]
    try:
        data = llm_client.chat_json(messages, temperature=0.3, max_tokens=256)
        scores: Dict[str, int | float] = {}
        for dim in DIMENSIONS:
            val = data.get(dim, 0)
            scores[dim] = max(1, min(10, int(val)))
        scores["average"] = round(sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
        return scores
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return {d: 0 for d in DIMENSIONS} | {"average": 0.0}
