from __future__ import annotations

import json
from pathlib import Path

from tests.evaluation.quality_runner import (
    BASELINE_REPORT_PATH,
    LATEST_REPORT_PATH,
    BENCHMARK_PATH,
    compute_quality_metrics,
    load_benchmark_cases,
    run,
)


def test_benchmark_source_exists() -> None:
    assert BENCHMARK_PATH.exists()


def test_compute_quality_metrics_deterministic_expected_as_predicted() -> None:
    cases = load_benchmark_cases()
    metrics = compute_quality_metrics(cases)

    assert metrics["case_count"] >= 120
    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["entity_f1"] == 1.0
    assert metrics["relation_precision"] == 1.0
    assert metrics["relation_recall"] == 1.0
    assert metrics["relation_f1"] == 1.0
    assert metrics["contradiction_rate"] == 0.0
    assert metrics["coreference_accuracy"] == 1.0
    assert metrics["intent_accuracy"] == 1.0
    assert metrics["consistency_rate"] == 1.0


def test_run_baseline_generates_reports() -> None:
    report = run(mode="baseline")
    assert report["mode"] == "baseline"

    assert BASELINE_REPORT_PATH.exists()
    assert LATEST_REPORT_PATH.exists()

    baseline_json = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    latest_json = json.loads(LATEST_REPORT_PATH.read_text(encoding="utf-8"))
    assert baseline_json["metrics"] == latest_json["metrics"]


def test_run_compare_against_baseline_generates_delta() -> None:
    run(mode="baseline")
    report = run(mode="compare", against="baseline")

    assert report["mode"] == "compare"
    assert "delta" in report
    assert report["delta"]["entity_f1"] == 0.0
    assert report["delta"]["relation_f1"] == 0.0
    assert report["delta"]["coreference_accuracy"] == 0.0
    assert report["delta"]["intent_accuracy"] == 0.0
    assert report["delta"]["consistency_rate"] == 0.0


def test_two_consecutive_baseline_runs_are_identical() -> None:
    report1 = run(mode="baseline")
    report2 = run(mode="baseline")
    assert report1["metrics"] == report2["metrics"]
