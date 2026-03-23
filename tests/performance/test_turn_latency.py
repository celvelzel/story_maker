"""Performance-focused regression tests for per-turn summary caching."""

from unittest.mock import patch

from src.engine.game_engine import GameEngine


def _chat_json_router(messages, **kwargs):
    text = str(messages).lower()
    if "option" in text:
        return {
            "options": [
                {"text": "Explore", "intent_hint": "explore", "risk_level": "low"},
                {"text": "Rest", "intent_hint": "rest", "risk_level": "low"},
                {"text": "Talk", "intent_hint": "dialogue", "risk_level": "low"},
            ]
        }
    if "consistency checker" in text or "contradiction" in text:
        return {"conflicts": []}
    return {
        "entities": [{"name": "Hero", "type": "person", "description": "", "status": {}, "state_changes": {}}],
        "relations": [],
    }


@patch("src.utils.api_client.llm_client")
def test_process_turn_reuses_summary_within_turn(mock_llm):
    mock_llm.chat.return_value = "Narrative output"
    mock_llm.chat_json.side_effect = _chat_json_router

    engine = GameEngine(auto_load_nlu=False)
    engine.state.add_narration("Opening")

    call_counter = {"count": 0}
    original_to_summary = engine.kg.to_summary

    def counted_summary(*args, **kwargs):
        call_counter["count"] += 1
        return original_to_summary(*args, **kwargs)

    with patch.object(engine.kg, "to_summary", side_effect=counted_summary):
        result = engine.process_turn("look around")

    assert result.story_text
    # 1x story gen, 1x conflict detector _llm_check (bypasses engine cache), 1x options (post-KG update)
    assert call_counter["count"] <= 3
