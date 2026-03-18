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

    def __init__(self, genre: str = "fantasy", intent_model_path: Optional[str] = None, auto_load_nlu: bool = True):
        self.genre = genre
        self.state = GameState()
        self.kg = KnowledgeGraph()

        # Evaluation tracking
        self.turn_conflict_counts: List[int] = []

        # NLU
        self.coref = CoreferenceResolver()
        self.intent_clf = IntentClassifier(model_path=intent_model_path)
        self.entity_ext = EntityExtractor()
        self.nlu_status: Dict[str, object] = {
            "coref_loaded": False,
            "intent_model_loaded": False,
            "intent_backend": "rule_fallback",
            "entity_model_loaded": False,
        }

        if auto_load_nlu:
            self._load_nlu_components()
        else:
            logger.info("NLU auto-load disabled. Engine will run with lazy/fallback behavior.")

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
        self.turn_conflict_counts = []

        story_text = self.story_gen.generate_opening(self.genre)
        self.state.add_narration(story_text)

        # Seed the KG from the opening
        self._apply_extraction(story_text)

        kg_summary = self.kg.to_summary()
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg.graph)

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
        raw_conflicts = self.conflict_det.check_all(story_text)
        conflicts = [c.get("description", str(c)) for c in raw_conflicts]
        self.turn_conflict_counts.append(len(conflicts))

        # 7. Option generation
        kg_summary = self.kg.to_summary()  # refresh after KG update
        options = self.option_gen.generate(story_text, kg_summary)
        kg_html = render_kg_html(self.kg.graph)

        nlu_debug = {
            "resolved_input": resolved,
            "intent": intent,
            "confidence": intent_result["confidence"],
            "entities": entities,
            "intent_backend": self.nlu_status["intent_backend"],
            "intent_model_loaded": self.nlu_status["intent_model_loaded"],
            "coref_loaded": self.nlu_status["coref_loaded"],
            "entity_model_loaded": self.nlu_status["entity_model_loaded"],
        }

        return TurnResult(
            story_text=story_text,
            options=options,
            nlu_debug=nlu_debug,
            kg_html=kg_html,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @property
    def all_story_texts(self) -> List[str]:
        """Return all narrator texts from the session (for evaluation)."""
        return [
            e["text"] for e in self.state.story_history if e["role"] == "narrator"
        ]

    @property
    def kg_entity_names(self) -> List[str]:
        """Return the display names of all KG entities."""
        return [
            data.get("name", key)
            for key, data in self.kg.graph.nodes(data=True)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_nlu_components(self) -> None:
        """Load all NLU submodules with graceful degradation on failure."""
        try:
            self.coref.load()
            self.nlu_status["coref_loaded"] = self.coref.model is not None
        except Exception as exc:
            logger.warning("Coreference resolver init failed: %s", exc)

        try:
            self.intent_clf.load()
            self.nlu_status["intent_model_loaded"] = (
                self.intent_clf.model is not None and self.intent_clf.tokenizer is not None
            )
            self.nlu_status["intent_backend"] = self.intent_clf.backend
        except Exception as exc:
            logger.warning("Intent classifier init failed: %s", exc)
            self.nlu_status["intent_backend"] = "rule_fallback"

        try:
            self.entity_ext.load()
            self.nlu_status["entity_model_loaded"] = self.entity_ext.nlp is not None
        except Exception as exc:
            logger.warning("Entity extractor init failed: %s", exc)

        logger.info(
            "NLU load status: coref_loaded=%s, intent_model_loaded=%s, intent_backend=%s, entity_model_loaded=%s",
            self.nlu_status["coref_loaded"],
            self.nlu_status["intent_model_loaded"],
            self.nlu_status["intent_backend"],
            self.nlu_status["entity_model_loaded"],
        )

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
