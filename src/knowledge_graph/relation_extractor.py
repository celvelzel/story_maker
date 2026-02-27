"""LLM-based entity + relation extraction from story text.

Sends text to the OpenAI API (via ``llm_client``) in JSON mode and returns
structured entities and relations for KG ingestion.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = (
    "You are a knowledge-graph extraction engine for a text-adventure game. "
    "Given a passage of story text, extract entities and relations.\n"
    "Return ONLY valid JSON with the schema:\n"
    '{"entities": [{"name": str, "type": "person"|"location"|"item"|"creature"|"event"}], '
    '"relations": [{"source": str, "target": str, "relation": str}]}\n'
    "Keep relation types simple and lowercase, e.g. located_at, possesses, ally_of, enemy_of, knows, etc."
)


class RelationExtractor:
    """Extract entities and relations from story text using the LLM."""

    def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return ``{"entities": [...], "relations": [...]}`` parsed from LLM JSON.

        On any failure an empty result is returned so the game never crashes.
        """
        try:
            from src.utils.api_client import llm_client

            messages = [
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": text},
            ]
            data = llm_client.chat_json(messages, temperature=0.2, max_tokens=512)
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            return {"entities": entities, "relations": relations}
        except Exception as exc:
            logger.warning("Relation extraction failed: %s", exc)
            return {"entities": [], "relations": []}


# Module-level convenience singleton + function
_extractor = RelationExtractor()


def extract(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Module-level shortcut for ``RelationExtractor().extract(text)``."""
    return _extractor.extract(text)
