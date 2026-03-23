from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from src.evaluation.metrics import (
    conflict_signature,
    consistency_rate,
    entity_signature,
    exact_match_accuracy,
    overlap_prf,
    relation_signature,
)


BENCHMARK_PATH = Path(__file__).parent / "data" / "nlu_kg_quality_benchmark.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"
LATEST_REPORT_PATH = REPORT_DIR / "latest_quality.json"
BASELINE_REPORT_PATH = REPORT_DIR / "baseline_quality.json"


@dataclass(frozen=True)
class CasePayload:
    id: str
    scenario_tag: str
    input: str
    prior_context: List[str]
    expected_entities: List[Dict[str, Any]]
    expected_relations: List[Dict[str, Any]]
    expected_conflicts: List[Dict[str, Any]]
    expected_intent: str


def load_benchmark_cases(path: Path = BENCHMARK_PATH) -> List[CasePayload]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    return [
        CasePayload(
            id=row["id"],
            scenario_tag=row["scenario_tag"],
            input=row["input"],
            prior_context=row["prior_context"],
            expected_entities=row["expected_entities"],
            expected_relations=row["expected_relations"],
            expected_conflicts=row["expected_conflicts"],
            expected_intent=row["expected_intent"],
        )
        for row in rows
    ]


def _entity_sigs(items: Iterable[Mapping[str, Any]]) -> List[str]:
    return [entity_signature(item.get("name", ""), item.get("type", "")) for item in items]


def _relation_sigs(items: Iterable[Mapping[str, Any]]) -> List[str]:
    return [
        relation_signature(item.get("source", ""), item.get("target", ""), item.get("relation", ""))
        for item in items
    ]


def _conflict_sigs(items: Iterable[Mapping[str, Any]]) -> List[str]:
    return [conflict_signature(item.get("type", ""), item.get("entity", "")) for item in items]


def _default_prediction(case: CasePayload) -> Dict[str, Any]:
    """Wave-A deterministic baseline prediction (expected-as-predicted)."""
    return {
        "entities": case.expected_entities,
        "relations": case.expected_relations,
        "conflicts": case.expected_conflicts,
        "intent": case.expected_intent,
    }


def compute_quality_metrics(
    cases: List[CasePayload],
    predictions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    pred_entities: List[str] = []
    exp_entities: List[str] = []
    pred_relations: List[str] = []
    exp_relations: List[str] = []
    intent_pred: List[str] = []
    intent_exp: List[str] = []

    unexpected_conflict_cases = 0
    turn_conflict_counts: List[int] = []

    pronoun_case_hits = 0
    pronoun_case_total = 0

    for case in cases:
        pred = dict(predictions_by_id.get(case.id, {})) if predictions_by_id else _default_prediction(case)

        p_entities = _entity_sigs(pred.get("entities", []))
        e_entities = _entity_sigs(case.expected_entities)
        pred_entities.extend(p_entities)
        exp_entities.extend(e_entities)

        p_relations = _relation_sigs(pred.get("relations", []))
        e_relations = _relation_sigs(case.expected_relations)
        pred_relations.extend(p_relations)
        exp_relations.extend(e_relations)

        p_conflicts = set(_conflict_sigs(pred.get("conflicts", [])))
        e_conflicts = set(_conflict_sigs(case.expected_conflicts))
        unexpected = p_conflicts - e_conflicts
        turn_conflict_counts.append(len(unexpected))
        if unexpected:
            unexpected_conflict_cases += 1

        intent_pred.append(str(pred.get("intent", "")).strip().lower())
        intent_exp.append(case.expected_intent.strip().lower())

        if case.scenario_tag == "pronoun_heavy_dialogue":
            pronoun_case_total += 1
            p_people = {sig for sig in p_entities if sig.endswith("::person")}
            e_people = {sig for sig in e_entities if sig.endswith("::person")}
            if p_people == e_people and e_people:
                pronoun_case_hits += 1

    entity_prf = overlap_prf(pred_entities, exp_entities)
    relation_prf = overlap_prf(pred_relations, exp_relations)
    intent_acc = exact_match_accuracy(intent_pred, intent_exp)

    contradiction_rate = (unexpected_conflict_cases / len(cases)) if cases else 0.0
    coref_accuracy = (pronoun_case_hits / pronoun_case_total) if pronoun_case_total else 1.0

    return {
        "case_count": len(cases),
        "entity_precision": entity_prf["precision"],
        "entity_recall": entity_prf["recall"],
        "entity_f1": entity_prf["f1"],
        "relation_precision": relation_prf["precision"],
        "relation_recall": relation_prf["recall"],
        "relation_f1": relation_prf["f1"],
        "contradiction_rate": contradiction_rate,
        "coreference_accuracy": coref_accuracy,
        "intent_accuracy": intent_acc,
        "consistency_rate": consistency_rate(turn_conflict_counts),
    }


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float)):
            delta[key] = float(value) - float(baseline[key])
    return delta


def evaluate_quality_gates(
    current_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    previous_gate_pass: bool = False,
) -> Dict[str, Any]:
    """Evaluate Wave-A Gate-1/2/3 policy.

    Gate-1: No tracked metric regresses more than 1.0 percentage point.
    Gate-2: relation_f1 +3pp, coreference_accuracy +5pp, contradiction_rate -20% relative.
    Gate-3: Gate-1+2 pass in two consecutive runs.
    """
    tracked = [
        "entity_f1",
        "relation_f1",
        "coreference_accuracy",
        "intent_accuracy",
        "consistency_rate",
    ]
    max_regression_pp = 0.01

    regressions: Dict[str, float] = {}
    for key in tracked:
        cur = float(current_metrics.get(key, 0.0))
        base = float(baseline_metrics.get(key, 0.0))
        diff = cur - base
        if diff < -max_regression_pp:
            regressions[key] = diff

    gate_1_pass = len(regressions) == 0

    relation_gain = float(current_metrics.get("relation_f1", 0.0)) - float(
        baseline_metrics.get("relation_f1", 0.0)
    )
    coref_gain = float(current_metrics.get("coreference_accuracy", 0.0)) - float(
        baseline_metrics.get("coreference_accuracy", 0.0)
    )
    base_contradiction = float(baseline_metrics.get("contradiction_rate", 0.0))
    current_contradiction = float(current_metrics.get("contradiction_rate", 0.0))

    if base_contradiction <= 0.0:
        contradiction_target_hit = current_contradiction <= 0.0
    else:
        contradiction_target_hit = current_contradiction <= base_contradiction * 0.8

    gate_2_target_pass = (
        relation_gain >= 0.03 and coref_gain >= 0.05 and contradiction_target_hit
    )

    current_pass = gate_1_pass and gate_2_target_pass
    gate_3_two_consecutive_pass = current_pass and previous_gate_pass

    return {
        "gate_1_pass": gate_1_pass,
        "gate_1_max_regression_pp": max_regression_pp,
        "gate_1_regressions": regressions,
        "gate_2_target_pass": gate_2_target_pass,
        "gate_2_relation_f1_gain": relation_gain,
        "gate_2_coreference_accuracy_gain": coref_gain,
        "gate_2_contradiction_target_hit": contradiction_target_hit,
        "gate_3_two_consecutive_pass": gate_3_two_consecutive_pass,
    }


def run(mode: str, against: str | None = None, out: Path | None = None) -> Dict[str, Any]:
    cases = load_benchmark_cases()
    current = compute_quality_metrics(cases)

    now = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "mode": mode,
        "generated_at": now,
        "benchmark_path": str(BENCHMARK_PATH),
        "metrics": current,
    }

    if mode == "baseline":
        _write_report(BASELINE_REPORT_PATH, report)
        _write_report(out or LATEST_REPORT_PATH, report)
        return report

    if mode == "compare":
        if against is None:
            raise ValueError("--against is required in compare mode")
        baseline_path = BASELINE_REPORT_PATH if against == "baseline" else Path(against)
        baseline = _load_report(baseline_path)
        baseline_metrics = baseline.get("metrics", {})
        delta = _build_delta(current, baseline_metrics)

        previous_gate_pass = False
        prior_report_path = out or LATEST_REPORT_PATH
        if prior_report_path.exists():
            try:
                prior_report = _load_report(prior_report_path)
                prev_summary = prior_report.get("gate_summary", {})
                previous_gate_pass = bool(
                    prev_summary.get("gate_1_pass", False)
                    and prev_summary.get("gate_2_target_pass", False)
                )
            except Exception:
                previous_gate_pass = False

        gate_summary = evaluate_quality_gates(
            current_metrics=current,
            baseline_metrics=baseline_metrics,
            previous_gate_pass=previous_gate_pass,
        )

        report["against"] = str(baseline_path)
        report["baseline_metrics"] = baseline_metrics
        report["delta"] = delta
        report["gate_summary"] = gate_summary
        _write_report(out or LATEST_REPORT_PATH, report)
        return report

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality benchmark runner")
    parser.add_argument("--mode", choices=["baseline", "compare"], required=True)
    parser.add_argument("--against", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    report = run(mode=args.mode, against=args.against, out=out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
