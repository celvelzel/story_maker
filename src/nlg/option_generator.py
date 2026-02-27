"""Player option generation via LLM with hardcoded fallback.

Each option carries ``text``, ``intent_hint``, and ``risk_level``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from config import settings
from src.nlg.prompt_templates import OPTION_GENERATION_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class StoryOption:
    """A single player choice."""
    text: str
    intent_hint: str = "other"
    risk_level: str = "medium"


# Hardcoded safety net
_FALLBACK_OPTIONS: List[StoryOption] = [
    StoryOption("Look around and assess the situation.", "explore", "low"),
    StoryOption("Move cautiously forward.", "action", "medium"),
    StoryOption("Try to speak with someone nearby.", "dialogue", "low"),
]


class OptionGenerator:
    """Generate contextual options for the player using the LLM."""

    def generate(
        self,
        story_text: str,
        kg_summary: str,
        num_options: int | None = None,
    ) -> List[StoryOption]:
        """Return a list of ``StoryOption``.  Falls back to defaults on error."""
        num_options = num_options or settings.NUM_OPTIONS
        try:
            from src.utils.api_client import llm_client

            user_msg = OPTION_GENERATION_PROMPT.format(
                num_options=num_options,
                story_text=story_text[-1500:],
                kg_summary=kg_summary[-1000:],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            data = llm_client.chat_json(messages, temperature=0.8, max_tokens=512)
            raw_options = data.get("options", [])
            options = [
                StoryOption(
                    text=o.get("text", "Continue…"),
                    intent_hint=o.get("intent_hint", "other"),
                    risk_level=o.get("risk_level", "medium"),
                )
                for o in raw_options
            ]
            if options:
                return options[:num_options]
        except Exception as exc:
            logger.warning("Option generation failed (%s) – using fallback.", exc)

        return list(_FALLBACK_OPTIONS[:num_options])
