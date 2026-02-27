"""Integration tests: full engine pipeline with mocked LLM calls."""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.state import GameState
from src.knowledge_graph.graph import KnowledgeGraph
from src.nlg.option_generator import StoryOption


# ── Shared mock helpers ─────────────────────────────────────────────

def _mock_llm():
    """Return a mock that covers chat() and chat_json()."""
    m = MagicMock()
    m.chat.return_value = "The adventure continues in a moonlit glade."
    m.chat_json.side_effect = _chat_json_router
    return m


def _chat_json_router(messages, **kwargs):
    """Route chat_json calls based on prompt content."""
    text = str(messages)
    if "option" in text.lower():
        return {
            "options": [
                {"text": "Explore the glade", "intent_hint": "explore", "risk_level": "low"},
                {"text": "Set up camp", "intent_hint": "rest", "risk_level": "low"},
                {"text": "Call out into the darkness", "intent_hint": "dialogue", "risk_level": "medium"},
            ]
        }
    # relation extraction
    return {
        "entities": [{"name": "Glade", "type": "location"}],
        "relations": [],
    }


# ── GameState unit tests ────────────────────────────────────────────

class TestGameState:
    def test_add_narration(self):
        s = GameState()
        s.add_narration("Once upon a time.")
        assert len(s.story_history) == 1
        assert s.story_history[0]["role"] == "narrator"

    def test_add_player_input(self):
        s = GameState()
        s.add_player_input("go north")
        assert s.story_history[0]["role"] == "player"

    def test_recent_history_limit(self):
        s = GameState()
        for i in range(10):
            s.add_narration(f"Event {i}")
        # recent_history() returns a formatted string, not a list
        assert len(s.recent_history(3).split("\n")) == 3


# ── Engine: start_game ──────────────────────────────────────────────

class TestEngineStartGame:
    @patch("src.nlg.story_generator.llm_client")
    @patch("src.nlg.option_generator.llm_client")
    @patch("src.knowledge_graph.relation_extractor.llm_client")
    def test_start_game_returns_turn_result(self, mock_re, mock_opt, mock_sg):
        mock_sg.chat.return_value = "You stand at the edge of a dark forest."
        mock_opt.chat_json.return_value = {
            "options": [
                {"text": "Enter the forest", "intent_hint": "explore", "risk_level": "medium"},
            ]
        }
        mock_re.chat_json.return_value = {"entities": [], "relations": []}

        engine = GameEngine(genre="fantasy")
        result = engine.start_game()

        assert isinstance(result, TurnResult)
        assert len(result.story_text) > 0
        assert len(result.options) >= 1


# ── Engine: process_turn ────────────────────────────────────────────

class TestEngineProcessTurn:
    @patch("src.nlg.story_generator.llm_client")
    @patch("src.nlg.option_generator.llm_client")
    @patch("src.knowledge_graph.relation_extractor.llm_client")
    @patch("src.knowledge_graph.conflict_detector.llm_client")
    def test_process_turn_returns_turn_result(
        self, mock_cd, mock_re, mock_opt, mock_sg
    ):
        mock_sg.chat.return_value = "The moonlit path unfolds before you."
        mock_opt.chat_json.return_value = {
            "options": [
                {"text": "Follow the path", "intent_hint": "explore", "risk_level": "low"},
            ]
        }
        mock_re.chat_json.return_value = {"entities": [], "relations": []}
        mock_cd.chat_json.return_value = {"conflicts": []}

        engine = GameEngine(genre="fantasy")
        # Seed state so process_turn has history
        engine.state.add_narration("You awaken in a glade.")

        result = engine.process_turn("look around")
        assert isinstance(result, TurnResult)
        assert "intent" in result.nlu_debug

    @patch("src.nlg.story_generator.llm_client")
    @patch("src.nlg.option_generator.llm_client")
    @patch("src.knowledge_graph.relation_extractor.llm_client")
    @patch("src.knowledge_graph.conflict_detector.llm_client")
    def test_two_turns_accumulate_state(
        self, mock_cd, mock_re, mock_opt, mock_sg
    ):
        mock_sg.chat.return_value = "Story text."
        mock_opt.chat_json.return_value = {"options": [{"text": "Continue"}]}
        mock_re.chat_json.return_value = {"entities": [], "relations": []}
        mock_cd.chat_json.return_value = {"conflicts": []}

        engine = GameEngine()
        engine.state.add_narration("Opening.")

        engine.process_turn("action one")
        engine.process_turn("action two")

        # 1 opening + 2*(player+narration) = 5
        assert len(engine.state.story_history) == 5


# ── KG integration ──────────────────────────────────────────────────

class TestKGIntegration:
    def test_entities_added_across_turns(self):
        kg = KnowledgeGraph()
        kg.add_entity("Hero", "person")
        kg.add_entity("Forest", "location")
        kg.add_relation("Hero", "Forest", "entered")
        assert kg.graph.number_of_nodes() == 2
        assert kg.graph.number_of_edges() == 1

        kg.add_entity("Dragon", "creature")
        kg.add_relation("Hero", "Dragon", "encountered")
        assert kg.graph.number_of_nodes() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
