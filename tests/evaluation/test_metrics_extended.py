from __future__ import annotations

from src.evaluation.metrics import (
    flesch_reading_ease,
    full_evaluation,
    graph_density_evolution,
    lexical_overlap,
    type_token_ratio,
)


def test_lexical_overlap_empty_is_zero() -> None:
    assert lexical_overlap("", "") == 0.0


def test_lexical_overlap_jaccard_basic() -> None:
    # {hero, explores, cave} vs {hero, enters, cave}
    score = lexical_overlap("Hero explores cave", "Hero enters cave")
    assert score == 0.5


def test_type_token_ratio_empty_and_repeated() -> None:
    assert type_token_ratio("") == 0.0
    assert type_token_ratio("hero hero cave") == 2 / 3


def test_flesch_reading_ease_empty_safe() -> None:
    assert flesch_reading_ease("") == 0.0


def test_flesch_reading_ease_returns_float() -> None:
    score = flesch_reading_ease("The hero opens the old wooden door.")
    assert isinstance(score, float)


def test_graph_density_evolution_handles_low_node_cases() -> None:
    result = graph_density_evolution(
        [
            {"turn_id": 1, "node_count": 1, "edge_count": 0},
            {"turn_id": 2, "node_count": 2, "edge_count": 1},
            {"turn_id": 3, "node_count": 3, "edge_count": 2},
        ]
    )
    assert result["density_start"] == 0.0
    assert result["density_end"] == 0.333333
    assert result["density_delta"] == 0.333333
    assert result["density_series"] == [0.0, 0.5, 0.333333]


def test_full_evaluation_contains_new_metric_keys() -> None:
    metrics = full_evaluation(
        texts=["Hero enters the cave.", "Hero lights a torch."],
        entity_names=["Hero", "Cave"],
        turn_conflict_counts=[0, 1],
        kg_turn_stats=[
            {"turn_id": 1, "node_count": 2, "edge_count": 1},
            {"turn_id": 2, "node_count": 3, "edge_count": 2},
        ],
    )
    assert "lexical_overlap" in metrics
    assert "type_token_ratio" in metrics
    assert "flesch_reading_ease" in metrics
    assert "graph_density_average" in metrics
    assert "graph_density_delta" in metrics
