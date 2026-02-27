"""Entity extraction: spaCy NER + noun-phrase heuristic.

Returns list of dicts: {"text", "type", "start", "end", "source"}.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Map spaCy NER labels to our game types
LABEL_MAP: Dict[str, str] = {
    "PERSON": "person",
    "ORG": "person",       # factions / groups → person
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "PRODUCT": "item",
    "WORK_OF_ART": "item",
}

# Heuristic word sets for _infer_type
_CREATURE_WORDS = {
    "dragon", "wolf", "goblin", "troll", "orc", "spider", "serpent",
    "bear", "beast", "demon", "wraith", "ghost", "undead", "skeleton",
    "rat", "bat", "griffin", "phoenix", "basilisk", "hydra", "wyvern",
}
_LOCATION_WORDS = {
    "forest", "cave", "castle", "village", "town", "city", "river",
    "mountain", "temple", "dungeon", "tower", "bridge", "lake", "desert",
    "island", "palace", "fortress", "ruins", "shrine", "tavern", "market",
    "harbor", "garden", "cemetery", "mine", "swamp", "valley", "cliff",
}
_ITEM_WORDS = {
    "sword", "shield", "potion", "scroll", "amulet", "ring", "staff",
    "bow", "arrow", "gem", "key", "map", "book", "dagger", "axe",
    "armor", "helmet", "cloak", "orb", "crystal", "relic", "artifact",
    "tome", "wand", "lantern", "rope", "torch", "coin", "chest",
}


class EntityExtractor:
    """spaCy NER + noun-phrase extraction with type inference."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model_name = spacy_model
        self.nlp = None

    def load(self) -> None:
        try:
            import spacy
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info("spaCy model loaded: %s", self.spacy_model_name)
        except Exception as exc:
            logger.warning("spaCy load failed (%s) – using noun-phrase only.", exc)
            self.nlp = None

    # ── public API ────────────────────────────────────────
    def extract(self, text: str) -> List[Dict[str, object]]:
        """Return a deduplicated list of entity dicts."""
        entities: List[Dict[str, object]] = []
        if self.nlp:
            entities.extend(self._spacy_extract(text))
        entities.extend(self._noun_phrase_extract(text))
        return self._deduplicate(entities)

    # ── spaCy NER ─────────────────────────────────────────
    def _spacy_extract(self, text: str) -> List[Dict[str, object]]:
        doc = self.nlp(text)  # type: ignore[union-attr]
        results: List[Dict[str, object]] = []
        for ent in doc.ents:
            game_type = LABEL_MAP.get(ent.label_)
            if game_type:
                results.append({
                    "text": ent.text,
                    "type": game_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy",
                })
        return results

    # ── noun phrase + heuristic type ──────────────────────
    def _noun_phrase_extract(self, text: str) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        if self.nlp:
            doc = self.nlp(text)  # type: ignore[union-attr]
            for chunk in doc.noun_chunks:
                # skip very short / pronoun chunks
                head = chunk.root.text.lower()
                if head in ("i", "you", "he", "she", "it", "we", "they", "me", "my"):
                    continue
                inferred = self._infer_type(head)
                if inferred:
                    results.append({
                        "text": chunk.text,
                        "type": inferred,
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "source": "noun_phrase",
                    })
        else:
            # Regex fallback for common patterns
            for word in re.findall(r"\b[A-Za-z]{3,}\b", text):
                inferred = self._infer_type(word.lower())
                if inferred:
                    start = text.lower().find(word.lower())
                    results.append({
                        "text": word,
                        "type": inferred,
                        "start": start,
                        "end": start + len(word),
                        "source": "regex",
                    })
        return results

    @staticmethod
    def _infer_type(word: str) -> Optional[str]:
        w = word.lower()
        if w in _CREATURE_WORDS:
            return "creature"
        if w in _LOCATION_WORDS:
            return "location"
        if w in _ITEM_WORDS:
            return "item"
        return None

    # ── dedup ─────────────────────────────────────────────
    @staticmethod
    def _deduplicate(entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen: Dict[str, Dict[str, object]] = {}
        for ent in entities:
            key = str(ent["text"]).lower()
            # prefer spacy source
            if key not in seen or ent.get("source") == "spacy":
                seen[key] = ent
        return list(seen.values())
