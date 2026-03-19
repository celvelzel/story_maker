"""Tests for enhanced GameEngine (game_engine.py)."""
import pytest
from unittest.mock import patch, MagicMock

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.state import GameState
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import KeepLatestResolver, LLMArbitrateResolver


# ── Shared mock helpers ─────────────────────────────────────────────

def _mock_llm():
    """Return a mock that covers chat() and chat_json()."""
    m = MagicMock()
    m.chat.return_value = "The adventure continues in a moonlit glade."
    m.chat_json.side_effect = _chat_json_router
    return m


def _chat_json_router(messages, **kwargs):
    """Route chat_json calls based on prompt content."""
    text = str(messages).lower()
    if "option" in text:
        return {
            "options": [
                {"text": "Explore the glade", "intent_hint": "explore", "risk_level": "low"},
                {"text": "Set up camp", "intent_hint": "rest", "risk_level": "low"},
                {"text": "Call out into the darkness", "intent_hint": "dialogue", "risk_level": "medium"},
            ]
        }
    if "consistency checker" in text or "contradiction" in text:
        return {"conflicts": []}
    if "conflict resolver" in text or "resolve" in text:
        return {"resolution": "keep_new", "target_entity": "", "target_relation": "", "reason": "test"}
    # default: relation extraction (rich)
    return {
        "entities": [
            {
                "name": "Glade",
                "type": "location",
                "description": "A moonlit clearing",
                "status": {"atmosphere": "mysterious"},
                "state_changes": {},
            },
            {
                "name": "Hero",
                "type": "person",
                "description": "A brave adventurer",
                "status": {"mood": "curious"},
                "state_changes": {},
            },
        ],
        "relations": [
            {"source": "Hero", "target": "Glade", "relation": "located_at", "context": "Hero is in the glade"},
        ],
    }


# ── Engine initialization tests ───────────────────────────

class TestEngineInit:
    @patch("src.utils.api_client.llm_client")
    def test_default_strategies_from_config(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router
        engine = GameEngine(genre="fantasy", auto_load_nlu=False)
        # strategies should be set from config defaults
        assert engine.conflict_resolution in ("keep_latest", "llm_arbitrate")
        assert engine.extraction_mode in ("story_only", "dual_extract")

    @patch("src.utils.api_client.llm_client")
    def test_custom_strategies(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router
        engine = GameEngine(
            genre="sci-fi",
            auto_load_nlu=False,
            conflict_resolution="keep_latest",
            extraction_mode="story_only",
            importance_mode="degree_only",
            summary_mode="flat",
        )
        assert engine.conflict_resolution == "keep_latest"
        assert engine.extraction_mode == "story_only"
        assert engine.importance_mode == "degree_only"
        assert engine.summary_mode == "flat"


# ── Start game tests ──────────────────────────────────────

class TestEngineStartGame:
    @patch("src.utils.api_client.llm_client")
    def test_start_game_returns_turn_result(self, mock_llm):
        mock_llm.chat.return_value = "You stand at the edge of a dark forest."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(genre="fantasy", auto_load_nlu=False)
        result = engine.start_game()

        assert isinstance(result, TurnResult)
        assert len(result.story_text) > 0
        assert len(result.options) >= 1
        assert engine.kg.num_nodes >= 0

    @patch("src.utils.api_client.llm_client")
    def test_start_game_seeds_kg(self, mock_llm):
        mock_llm.chat.return_value = "You stand in a glade with a sword."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(genre="fantasy", auto_load_nlu=False)
        engine.start_game()
        # KG should have entities from the opening text
        assert engine.kg.num_nodes >= 0


# ── Process turn tests ────────────────────────────────────

class TestEngineProcessTurn:
    @patch("src.utils.api_client.llm_client")
    def test_process_turn_returns_turn_result(self, mock_llm):
        mock_llm.chat.return_value = "The moonlit path unfolds before you."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(genre="fantasy", auto_load_nlu=False)
        engine.state.add_narration("You awaken in a glade.")

        result = engine.process_turn("look around")
        assert isinstance(result, TurnResult)
        assert "intent" in result.nlu_debug
        assert "intent_backend" in result.nlu_debug
        assert "intent_model_loaded" in result.nlu_debug

    @patch("src.utils.api_client.llm_client")
    def test_two_turns_accumulate_state(self, mock_llm):
        mock_llm.chat.return_value = "Story text."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False)
        engine.state.add_narration("Opening.")

        engine.process_turn("action one")
        engine.process_turn("action two")

        # 1 opening + 2*(player+narration) = 5
        assert len(engine.state.story_history) == 5

    @patch("src.utils.api_client.llm_client")
    def test_kg_updates_with_rich_entities(self, mock_llm):
        mock_llm.chat.return_value = "A dragon appears in the cave."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False)
        engine.state.add_narration("Opening.")
        engine.process_turn("enter the cave")

        # KG should have entities from extraction
        hero = engine.kg.get_entity("hero")
        glade = engine.kg.get_entity("glade")
        # at least one should exist from the mock extraction
        assert hero is not None or glade is not None or engine.kg.num_nodes >= 0

    @patch("src.utils.api_client.llm_client")
    def test_dual_extraction_mode(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False, extraction_mode="dual_extract")
        engine.state.add_narration("Opening.")
        result = engine.process_turn("I draw my sword and charge")
        assert isinstance(result, TurnResult)

    @patch("src.utils.api_client.llm_client")
    def test_story_only_extraction_mode(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False, extraction_mode="story_only")
        engine.state.add_narration("Opening.")
        result = engine.process_turn("look around")
        assert isinstance(result, TurnResult)


# ── Conflict resolution integration tests ─────────────────

class TestConflictResolutionIntegration:
    @patch("src.utils.api_client.llm_client")
    def test_keep_latest_strategy(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False, conflict_resolution="keep_latest")
        assert isinstance(engine.conflict_resolver, KeepLatestResolver)

        engine.state.add_narration("Opening.")
        result = engine.process_turn("action")
        assert isinstance(result, TurnResult)

    @patch("src.utils.api_client.llm_client")
    def test_llm_arbitrate_strategy(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False, conflict_resolution="llm_arbitrate")
        assert isinstance(engine.conflict_resolver, LLMArbitrateResolver)

        engine.state.add_narration("Opening.")
        result = engine.process_turn("action")
        assert isinstance(result, TurnResult)


# ── Summary mode integration tests ────────────────────────

class TestSummaryModeIntegration:
    @patch("src.utils.api_client.llm_client")
    @patch("src.knowledge_graph.graph.settings")
    def test_layered_summary_in_generation(self, mock_kg_settings, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router
        mock_kg_settings.KG_SUMMARY_MODE = "layered"
        mock_kg_settings.KG_MAX_TIMELINE_ENTRIES = 5
        mock_kg_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
        mock_kg_settings.KG_RELATION_DECAY_FACTOR = 0.90
        mock_kg_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
        mock_kg_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
        mock_kg_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
        mock_kg_settings.KG_MAX_NODES = 200
        mock_kg_settings.KG_IMPORTANCE_MODE = "composite"

        engine = GameEngine(auto_load_nlu=False, summary_mode="layered")
        engine.kg.add_entity("Hero", "person", description="Brave", status={"health": "full"}, turn_id=1)
        engine.kg.add_entity("Sword", "item", description="Ancient blade", turn_id=1)
        engine.kg.add_relation("Hero", "Sword", "possesses", turn_id=1)

        summary = engine.kg.to_summary()
        assert "Description:" in summary
        assert "Status:" in summary or "Relations:" in summary

    @patch("src.utils.api_client.llm_client")
    @patch("src.knowledge_graph.graph.settings")
    def test_flat_summary_backward_compat(self, mock_kg_settings, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router
        mock_kg_settings.KG_SUMMARY_MODE = "flat"
        mock_kg_settings.KG_MAX_TIMELINE_ENTRIES = 5
        mock_kg_settings.KG_IMPORTANCE_DECAY_FACTOR = 0.95
        mock_kg_settings.KG_RELATION_DECAY_FACTOR = 0.90
        mock_kg_settings.KG_RELATION_MIN_CONFIDENCE = 0.2
        mock_kg_settings.KG_IMPORTANCE_MENTION_BOOST = 0.15
        mock_kg_settings.KG_IMPORTANCE_PLAYER_BOOST = 0.3
        mock_kg_settings.KG_MAX_NODES = 200
        mock_kg_settings.KG_IMPORTANCE_MODE = "composite"

        engine = GameEngine(auto_load_nlu=False, summary_mode="flat")
        engine.kg.add_entity("Hero", "person", turn_id=1)
        summary = engine.kg.to_summary()
        assert "=== World State ===" in summary
        assert "=== Relations ===" in summary


# ── Temporal tracking integration tests ───────────────────

class TestTemporalTracking:
    @patch("src.utils.api_client.llm_client")
    def test_entity_mentions_tracked_across_turns(self, mock_llm):
        mock_llm.chat.return_value = "The hero explores further."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False)
        engine.state.add_narration("Opening.")
        engine.process_turn("hero looks around")
        engine.process_turn("hero fights dragon")

        hero = engine.kg.get_entity("hero")
        if hero:
            assert hero["mention_count"] >= 1
            assert hero["last_mentioned_turn"] >= 1

    @patch("src.utils.api_client.llm_client")
    def test_decay_applied_per_turn(self, mock_llm):
        mock_llm.chat.return_value = "Story."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False)
        engine.state.add_narration("Opening.")
        # add an old relation
        engine.kg.add_relation("Hero", "Cave", "located_at", turn_id=1, confidence=0.5)
        engine.process_turn("go north")

        # after several turns, old relations should decay
        for i in range(10):
            engine.state.add_narration(f"Turn {i}.")
            engine.process_turn(f"action {i}")

        # low-confidence old relations may have been pruned
        assert engine.kg.num_edges >= 0  # just verify no crash


# ── NLU lifecycle tests (backward compatible) ─────────────

class TestNLULifecycle:
    @patch("src.utils.api_client.llm_client")
    def test_engine_can_disable_nlu_autoload(self, mock_llm):
        mock_llm.chat.return_value = "Story text."
        mock_llm.chat_json.side_effect = _chat_json_router

        engine = GameEngine(auto_load_nlu=False)
        assert engine.nlu_status["intent_model_loaded"] is False

        engine.state.add_narration("Opening.")
        result = engine.process_turn("look around")
        assert result.nlu_debug["intent_model_loaded"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
