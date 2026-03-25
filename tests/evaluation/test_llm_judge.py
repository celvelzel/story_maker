from __future__ import annotations

from unittest.mock import patch

from src.evaluation.llm_judge import DIMENSIONS, judge


@patch("src.utils.api_client.llm_client")
def test_judge_parses_all_8_dimensions(mock_llm) -> None:
    mock_llm.chat_json.return_value = {
        "narrative_quality": 8,
        "consistency": 7,
        "player_agency": 9,
        "creativity": 8,
        "pacing": 7,
        "option_relevance": 8,
        "causal_link": 9,
        "local_coherence": 8,
    }
    result = judge("[Player] look around\n[Narrator] You see a cave.")

    for dim in DIMENSIONS:
        assert dim in result
        assert isinstance(result[dim], int)
        assert 1 <= result[dim] <= 10
    assert "average" in result
    assert result["average"] == round(sum(result[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)


@patch("src.utils.api_client.llm_client")
def test_judge_parses_json_string_with_noise(mock_llm) -> None:
    mock_llm.chat_json.return_value = (
        "Here is your result:\n"
        "{\"narrative_quality\": 9, \"consistency\": 8, \"player_agency\": 7, \"creativity\": 8, "
        "\"pacing\": 7, \"option_relevance\": 8, \"causal_link\": 7, \"local_coherence\": 8}\n"
        "Thank you."
    )
    result = judge("short transcript")
    assert result["narrative_quality"] == 9
    assert result["local_coherence"] == 8


@patch("src.utils.api_client.llm_client")
def test_judge_returns_zero_scores_on_exception(mock_llm) -> None:
    mock_llm.chat_json.side_effect = RuntimeError("network error")
    result = judge("short transcript")
    for dim in DIMENSIONS:
        assert result[dim] == 0
    assert result["average"] == 0.0
