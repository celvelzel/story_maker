from __future__ import annotations

from tests.evaluation.quality_runner import evaluate_quality_gates


def test_gate_1_rejects_regression_larger_than_1pp() -> None:
    baseline = {
        "entity_f1": 0.90,
        "relation_f1": 0.80,
        "coreference_accuracy": 0.70,
        "intent_accuracy": 0.95,
        "consistency_rate": 0.88,
        "contradiction_rate": 0.25,
    }
    current = {
        "entity_f1": 0.885,  # -1.5pp
        "relation_f1": 0.80,
        "coreference_accuracy": 0.70,
        "intent_accuracy": 0.95,
        "consistency_rate": 0.88,
        "contradiction_rate": 0.25,
    }
    summary = evaluate_quality_gates(current, baseline, previous_gate_pass=False)
    assert summary["gate_1_pass"] is False
    assert "entity_f1" in summary["gate_1_regressions"]


def test_gate_2_requires_three_targets() -> None:
    baseline = {
        "entity_f1": 0.90,
        "relation_f1": 0.60,
        "coreference_accuracy": 0.50,
        "intent_accuracy": 0.90,
        "consistency_rate": 0.85,
        "contradiction_rate": 0.20,
    }
    current = {
        "entity_f1": 0.91,
        "relation_f1": 0.63,  # +3pp
        "coreference_accuracy": 0.55,  # +5pp
        "intent_accuracy": 0.91,
        "consistency_rate": 0.86,
        "contradiction_rate": 0.16,  # -20%
    }
    summary = evaluate_quality_gates(current, baseline, previous_gate_pass=False)
    assert summary["gate_2_target_pass"] is True


def test_gate_3_requires_two_consecutive_passes() -> None:
    baseline = {
        "entity_f1": 0.90,
        "relation_f1": 0.60,
        "coreference_accuracy": 0.50,
        "intent_accuracy": 0.90,
        "consistency_rate": 0.85,
        "contradiction_rate": 0.20,
    }
    current = {
        "entity_f1": 0.91,
        "relation_f1": 0.63,
        "coreference_accuracy": 0.55,
        "intent_accuracy": 0.91,
        "consistency_rate": 0.86,
        "contradiction_rate": 0.16,
    }

    summary_first = evaluate_quality_gates(current, baseline, previous_gate_pass=False)
    assert summary_first["gate_3_two_consecutive_pass"] is False

    summary_second = evaluate_quality_gates(current, baseline, previous_gate_pass=True)
    assert summary_second["gate_3_two_consecutive_pass"] is True
