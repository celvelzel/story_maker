"""Entity extraction: spaCy NER + noun-phrase heuristic + KG context.

实体提取模块：结合 spaCy NER + 名词短语启发式 + 知识图谱上下文。

增强功能：
- 扩展词表（60+ 生物、50+ 地点、50+ 物品、魔法词汇）
- 知识图谱上下文辅助提取（与已知实体进行模糊匹配）
- 多词实体保留
- 所有格处理（"dragon's lair" → dragon）

返回字典列表：{"text", "type", "start", "end", "source"}。
"""
from __future__ import annotations

import logging
import importlib
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from config import settings

logger = logging.getLogger(__name__)

_ENTITY_FUZZY_THRESHOLDS: Dict[str, float] = {
    "person": 0.90,
    "location": 0.84,
    "creature": 0.84,
    "item": 0.80,
    "event": 0.88,
    "unknown": 0.86,
}

# Map spaCy NER labels to our game types (validated against KG_ENTITY_TYPES)
_LABEL_MAP_RAW: Dict[str, str] = {
    "PERSON": "person",
    "ORG": "person",       # factions / groups → person
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "PRODUCT": "item",
    "WORK_OF_ART": "item",
}

LABEL_MAP: Dict[str, str] = {}
for _spacy_label, _game_type in _LABEL_MAP_RAW.items():
    if _game_type in settings.KG_ENTITY_TYPES:
        LABEL_MAP[_spacy_label] = _game_type
    else:
        logger.warning(
            "LABEL_MAP value '%s' not in KG_ENTITY_TYPES, mapping '%s' -> 'unknown'",
            _game_type, _spacy_label,
        )
        LABEL_MAP[_spacy_label] = "unknown"

# ── Expanded heuristic word sets ──────────────────────────

_CREATURE_WORDS: Set[str] = {
    # Original (27)
    "dragon", "wolf", "goblin", "troll", "orc", "spider", "serpent",
    "bear", "beast", "demon", "wraith", "ghost", "undead", "skeleton",
    "rat", "bat", "griffin", "phoenix", "basilisk", "hydra", "wyvern",
    # Expanded fantasy creatures (35+)
    "zombie", "vampire", "werewolf", "lich", "imp", "djinn", "elemental",
    "golem", "harpy", "manticore", "chimera", "minotaur", "cyclops",
    "centaur", "fairy", "elf", "dwarf", "hobbit", "pixie", "sprite",
    "nymph", "dryad", "ogre", "gnoll", "kobold", "bugbear", "hobgoblin",
    "wyrm", "drake", "cockatrice", "beholder", "displacer", "owlbear",
    "horse", "eagle", "hawk", "snake", "lion", "tiger", "fox", "deer",
    "warrior", "knight", "mage", "archer", "rogue", "paladin", "cleric",
}

_LOCATION_WORDS: Set[str] = {
    # Original (27)
    "forest", "cave", "castle", "village", "town", "city", "river",
    "mountain", "temple", "dungeon", "tower", "bridge", "lake", "desert",
    "island", "palace", "fortress", "ruins", "shrine", "tavern", "market",
    "harbor", "garden", "cemetery", "mine", "swamp", "valley", "cliff",
    # Expanded (25+)
    "lair", "den", "pit", "abyss", "chasm", "cavern", "grotto", "hollow",
    "plains", "meadow", "field", "ocean", "sea", "bay", "port", "docks",
    "camp", "campsite", "outpost", "fort", "barracks", "academy", "library",
    "church", "sanctuary", "cathedral", "crypt", "tomb", "catacombs",
    "sewer", "passage", "corridor", "hall", "chamber", "throne",
    "kingdom", "realm", "domain", "territory", "land", "world",
}

_ITEM_WORDS: Set[str] = {
    # Original (30)
    "sword", "shield", "potion", "scroll", "amulet", "ring", "staff",
    "bow", "arrow", "gem", "key", "map", "book", "dagger", "axe",
    "armor", "helmet", "cloak", "orb", "crystal", "relic", "artifact",
    "tome", "wand", "lantern", "rope", "torch", "coin", "chest",
    # Expanded (25+)
    "blade", "spear", "mace", "flail", "halberd", "crossbow", "bolt",
    "quiver", "gauntlet", "greaves", "boots", "belt", "crown", "tiara",
    "necklace", "bracelet", "brooch", "pendant", "chalice", "goblet",
    "vial", "flask", "bottle", "bag", "sack", "pack", "crate", "box",
    "mirror", "compass", "hourglass", "instrument", "lute", "harp",
    "letter", "note", "diary", "journal", "contract", "deed",
    "food", "bread", "meat", "water", "wine", "ale", "herb", "flower",
    "seed", "stone", "rock", "boulder", "log", "plank", "nail", "chain",
    "rope", "hook", "lock", "trap", "bomb", "mine", "signal", "flag",
}

_MAGIC_WORDS: Set[str] = {
    "fireball", "lightning", "teleport", "heal", "curse", "charm",
    "shield", "barrier", "ward", "illusion", "invisibility", "levitate",
    "summon", "banish", "enchant", "disenchant", "dispel", "silence",
    "sleep", "fear", "haste", "slow", "polymorph", "transform",
    "spell", "incantation", "ritistic", "hex", "jinx", "charm",
    "blessing", "aura", "sigil", "glyph", "rune", "seal",
}

# All combined for quick lookup
_ALL_TYPED_WORDS: Dict[str, str] = {}
for _w in _CREATURE_WORDS:
    _ALL_TYPED_WORDS[_w] = "creature"
for _w in _LOCATION_WORDS:
    _ALL_TYPED_WORDS[_w] = "location"
for _w in _ITEM_WORDS:
    _ALL_TYPED_WORDS[_w] = "item"
for _w in _MAGIC_WORDS:
    _ALL_TYPED_WORDS[_w] = "item"  # Magic items → item type


class EntityExtractor:
    """spaCy NER + noun-phrase extraction with type inference and KG context.
    
    实体提取器：结合 spaCy NER + 名词短语提取 + 类型推断 + 知识图谱上下文。
    支持多种提取来源：spaCy NER、名词短语、启发式词表匹配。
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """
        初始化实体提取器。
        
        参数:
            spacy_model: spaCy 模型名称
        """
        self.spacy_model_name = spacy_model  # spaCy 模型名称
        self.nlp = None  # spaCy NLP 管道

    def load(self) -> None:
        """加载 spaCy 模型。如果加载失败，仅使用名词短语提取。"""
        try:
            spacy_module = importlib.import_module("spacy")
            self.nlp = spacy_module.load(self.spacy_model_name)
            logger.info("spaCy model loaded: %s", self.spacy_model_name)
        except Exception as exc:
            logger.warning("spaCy load failed (%s) – using noun-phrase only.", exc)
            self.nlp = None

    # ── public API ────────────────────────────────────────
    def extract(
        self,
        text: str,
        known_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        """Return a deduplicated list of entity dicts.

        提取文本中的实体，返回去重后的实体字典列表。
        
        参数:
            text: 要提取实体的文本
            known_entities: 知识图谱中已知实体名称列表（可选），
                用于模糊匹配和类型推断
        
        返回:
            List[Dict]: 实体字典列表，每个包含 text, type, start, end, source
        """
        entities: List[Dict[str, object]] = []
        if self.nlp:
            entities.extend(self._spacy_extract(text))
        entities.extend(self._noun_phrase_extract(text))

        # KG context-assisted enrichment
        if known_entities:
            entities = self._enrich_with_kg_context(entities, known_entities, text)

        deduped = self._deduplicate(entities)
        for ent in deduped:
            ent.setdefault("confidence", 0.7)
        return deduped

    # ── spaCy NER ─────────────────────────────────────────
    def _spacy_extract(self, text: str) -> List[Dict[str, object]]:
        """使用 spaCy NER 提取实体。"""
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
                    "confidence": 0.95,
                })
        return results

    # ── noun phrase + heuristic type ──────────────────────
    def _noun_phrase_extract(self, text: str) -> List[Dict[str, object]]:
        """使用名词短语提取 + 启发式类型推断。"""
        results: List[Dict[str, object]] = []
        if self.nlp:
            doc = self.nlp(text)  # type: ignore[union-attr]
            for chunk in doc.noun_chunks:
                # skip very short / pronoun chunks
                head = chunk.root.text.lower()
                if head in ("i", "you", "he", "she", "it", "we", "they", "me", "my"):
                    continue

                # Handle possessives: "dragon's lair" → extract "dragon"
                chunk_text = chunk.text
                possessive_match = re.match(r"^(\w+)'s\s+", chunk_text)
                if possessive_match:
                    possessor = possessive_match.group(1)
                    possessor_type = self._infer_type(possessor)
                    if possessor_type:
                        results.append({
                            "text": possessor,
                            "type": possessor_type,
                            "start": chunk.start_char,
                            "end": chunk.start_char + len(possessor),
                            "source": "possessive",
                            "confidence": 0.82,
                        })

                # Infer type from head word or entire phrase
                inferred = self._infer_type(head)
                if not inferred:
                    # Try inferring from the whole chunk text
                    inferred = self._infer_chunk_type(chunk.text)

                if inferred:
                    results.append({
                        "text": chunk.text,
                        "type": inferred,
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "source": "noun_phrase",
                        "confidence": 0.78,
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
                        "confidence": 0.70,
                    })

            # Proper-noun fallback for alias linking when spaCy is unavailable
            for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
                phrase = match.group(1)
                if phrase.lower() in {"the", "a", "an"}:
                    continue
                results.append({
                    "text": phrase,
                    "type": "unknown",
                    "start": match.start(1),
                    "end": match.end(1),
                    "source": "regex_proper_noun",
                    "confidence": 0.66,
                })
        return results

    @staticmethod
    def _normalize_alias(text: str) -> str:
        """Normalize entity mention for robust alias matching."""
        lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        lowered = re.sub(r"\b(the|a|an|sir|lady|lord|queen|king|captain)\b", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _entity_threshold(self, ent_type: str) -> float:
        return _ENTITY_FUZZY_THRESHOLDS.get(ent_type, _ENTITY_FUZZY_THRESHOLDS["unknown"])

    def _infer_type(self, word: str) -> Optional[str]:
        """Infer entity type from a single word."""
        return _ALL_TYPED_WORDS.get(word.lower())

    def _infer_chunk_type(self, chunk_text: str) -> Optional[str]:
        """Infer entity type from a noun phrase chunk by checking all words."""
        words = chunk_text.lower().split()
        # Check if any word in the chunk matches a known type
        # Prefer more specific types: creature > location > item
        for word in words:
            w = word.strip("'s")
            if w in _CREATURE_WORDS:
                return "creature"
        for word in words:
            w = word.strip("'s")
            if w in _LOCATION_WORDS:
                return "location"
        for word in words:
            w = word.strip("'s")
            if w in _ITEM_WORDS:
                return "item"
        return None

    # ── KG context enrichment ─────────────────────────────
    def _enrich_with_kg_context(
        self,
        entities: List[Dict[str, object]],
        known_entities: List[str],
        original_text: str = "",
    ) -> List[Dict[str, object]]:
        """Enrich extracted entities by fuzzy-matching against KG entities.

        - If an extracted entity fuzzy-matches a known entity, use the known name.
        - If the original text mentions a known entity that wasn't extracted, add it.
        """
        if not known_entities:
            return entities

        # Build set of already extracted names (lowered)
        extracted_names: Set[str] = {str(e["text"]).lower() for e in entities}
        known_lower_map: Dict[str, str] = {k.lower(): k for k in known_entities}
        known_norm_map: Dict[str, str] = {}
        for name in known_entities:
            norm = self._normalize_alias(name)
            if norm:
                known_norm_map[norm] = name

        # Enrich existing entities via fuzzy matching
        for ent in entities:
            ent_name = str(ent["text"]).lower()
            ent_norm = self._normalize_alias(ent_name)
            ent_type = str(ent.get("type", "unknown"))
            threshold = self._entity_threshold(ent_type)

            best_match, score = self._fuzzy_match_with_score(
                ent_norm,
                list(known_norm_map.keys()),
                threshold=threshold,
            )
            if best_match and best_match != ent_name:
                ent["text"] = known_norm_map[best_match]
                ent["source"] = str(ent.get("source", "")) + "+kg"
                ent["confidence"] = max(self._as_float(ent.get("confidence", 0.0)), score)

        text_lower = original_text.lower()

        # Alias mention scan: allow partial known-entity mention in text (e.g., "Gandalf" for "Gandalf the Grey")
        normalized_text = self._normalize_alias(original_text)
        text_tokens = set(normalized_text.split())
        for known_name in known_entities:
            if known_name.lower() in extracted_names:
                continue
            known_norm = self._normalize_alias(known_name)
            known_tokens = [token for token in known_norm.split() if len(token) >= 4]
            if not known_tokens:
                continue
            overlap = sum(1 for token in known_tokens if token in text_tokens)
            if overlap >= 1:
                pos = text_lower.find(known_tokens[0]) if known_tokens[0] in text_lower else -1
                entities.append({
                    "text": known_name,
                    "type": "unknown",
                    "start": max(pos, 0),
                    "end": max(pos, 0) + len(known_tokens[0]),
                    "source": "kg_alias",
                    "confidence": 0.74,
                })
                extracted_names.add(known_name.lower())

        # Scan original text for known entities that weren't extracted
        for known_name in known_entities:
            known_lower = known_name.lower()
            if known_lower in text_lower and known_lower not in extracted_names:
                pos = text_lower.find(known_lower)
                entities.append({
                    "text": known_name,
                    "type": "unknown",
                    "start": pos,
                    "end": pos + len(known_name),
                    "source": "kg_context",
                    "confidence": 0.72,
                })

        return entities

    @staticmethod
    def _fuzzy_match_with_score(
        query: str,
        candidates: List[str],
        threshold: float = 0.8,
    ) -> Tuple[Optional[str], float]:
        """Find best fuzzy match and score for query in candidates."""
        if not query or not candidates:
            return None, 0.0
        best_score = 0.0
        best_match: Optional[str] = None
        for candidate in candidates:
            score = SequenceMatcher(None, query, candidate).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        return best_match, best_score

    @staticmethod
    def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
        """Find the best fuzzy match for query in candidates."""
        best_match, _ = EntityExtractor._fuzzy_match_with_score(query, candidates, threshold)
        return best_match

    @staticmethod
    def _as_float(value: object, default: float = 0.0) -> float:
        try:
            if isinstance(value, (int, float, str)):
                return float(value)
            return default
        except (TypeError, ValueError):
            return default

    # ── dedup ─────────────────────────────────────────────
    @staticmethod
    def _deduplicate(entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen: Dict[str, Dict[str, object]] = {}
        for ent in entities:
            key = str(ent["text"]).lower()
            # prefer spacy source
            existing_source = str(seen.get(key, {}).get("source", ""))
            new_source = str(ent.get("source", ""))
            if key not in seen or "spacy" in new_source:
                seen[key] = ent
            elif "kg_context" in new_source and "spacy" not in existing_source:
                # Merge kg_context info into existing entity
                existing = seen[key]
                if existing.get("type") == "unknown" and ent.get("type") != "unknown":
                    existing["type"] = ent["type"]
        return list(seen.values())
