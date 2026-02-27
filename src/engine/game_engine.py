"""Main game engine orchestrator for StoryWeaver.

Pipeline per turn:
1. Coreference resolution
2. Intent classification
3. Entity extraction
4. Story generation (LLM)
5. KG update (relation extraction)
6. Conflict detection
7. Option generation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.engine.state import GameState
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.nlu.coreference import CoreferenceResolver
from src.nlg.story_generator import StoryGenerator
from src.nlg.option_generator import OptionGenerator, StoryOption
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.relation_extractor import extract as kg_extract
from src.knowledge_graph.conflict_detector import ConflictDetector
from src.knowledge_graph.visualizer import render_kg_html

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Container returned after every game turn."""
    story_text: str
    options: List[StoryOption]
    nlu_debug: Dict = field(default_factory=dict)
    kg_html: str = ""
    conflicts: List[str] = field(default_factory=list)


class GameEngine:
    """Coordinates the full NLU → NLG → KG pipeline."""

    def __init__(self, genre: str = "fantasy"):
        self.genre = genre
        self.state = GameState()
        self.kg = KnowledgeGraph()

        # NLU
        self.coref = CoreferenceResolver()
        self.intent_clf = IntentClassifier()
        self.entity_ext = EntityExtractor()

        # NLG
        self.story_gen = StoryGenerator()
        self.option_gen = OptionGenerator()

        # KG helpers
        self.conflict_det = ConflictDetector(self.kg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> TurnResult:
        """Begin a new adventure and return the opening turn."""
        self.state = GameState()
        self.kg = KnowledgeGraph()
        self.conflict_det = ConflictDetector(self.kg)

        story_text = self.story_gen.generate_opening(self.genre)
        self.state.add_narration(story_text)

        # Seed the KG from the opening
        self._apply_extraction(story_text)

        kg_summary = self.kg.to_summary()
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg)

        return TurnResult(
            story_text=story_text,
            options=options,
            kg_html=kg_html,
        )

    def process_turn(self, player_input: str) -> TurnResult:
        """Run the full pipeline for one player turn."""

        # 1. Coreference resolution
        recent_entries = self.state.story_history[-4:]
        recent_texts = [t["text"] for t in recent_entries]
        resolved = self.coref.resolve(player_input, recent_texts)

        # 2. Intent classification
        intent_result = self.intent_clf.predict(resolved)
        intent = intent_result["intent"]

        # 3. Entity extraction
        entities = self.entity_ext.extract(resolved)

        # Record player input
        self.state.add_player_input(player_input)

        # 4. Story generation
        kg_summary = self.kg.to_summary()
        history = self.state.recent_history(6)
        story_text = self.story_gen.continue_story(
            player_input=resolved,
            intent=intent,
            kg_summary=kg_summary,
            history=history,
        )
        self.state.add_narration(story_text)

        # 5. KG update from player entities + generated text
        for ent in entities:
            self.kg.add_entity(ent["text"], ent.get("type", "thing"))
        self._apply_extraction(story_text)

        # 6. Conflict detection
        conflicts = self.conflict_det.check_all(story_text)

        # 7. Option generation
        kg_summary = self.kg.to_summary()  # refresh after KG update
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg)

        nlu_debug = {
            "resolved_input": resolved,
            "intent": intent,
            "confidence": intent_result["confidence"],
            "entities": entities,
        }

        return TurnResult(
            story_text=story_text,
            options=options,
            nlu_debug=nlu_debug,
            kg_html=kg_html,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_extraction(self, text: str) -> None:
        """Run the LLM relation extractor and feed results into the KG."""
        try:
            data = kg_extract(text)
            for ent in data.get("entities", []):
                name = ent.get("name") or ent.get("text", "")
                etype = ent.get("type", "thing")
                if name:
                    self.kg.add_entity(name, etype)
            for rel in data.get("relations", []):
                src = rel.get("source", "")
                tgt = rel.get("target", "")
                label = rel.get("relation", "related_to")
                if src and tgt:
                    self.kg.add_relation(src, tgt, label)
        except Exception as exc:
            logger.warning("KG extraction failed: %s", exc)
