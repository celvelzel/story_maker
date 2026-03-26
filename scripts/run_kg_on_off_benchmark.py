"""Compare StoryWeaver performance/quality with KG enabled vs disabled.

Default experiment profile:
- genre: fantasy
- runs per condition: 1
- turns per run: 5
- policy: always pick the first generated option

Compared outputs:
- latency stats from GameEngine.process_turn
- automatic metrics from src.evaluation.metrics.full_evaluation
- LLM judge scores from src.evaluation.llm_judge.judge
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AUTO_KEYS = [
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "self_bleu",
    "entity_coverage",
    "consistency_rate",
    "type_token_ratio",
    "flesch_reading_ease",
    "lexical_overlap",
    "graph_density_average",
    "graph_density_delta",
]

DIMENSIONS = [
    "narrative_quality",
    "consistency",
    "player_agency",
    "creativity",
    "pacing",
    "option_relevance",
    "causal_link",
    "local_coherence",
]


@dataclass
class SessionResult:
    session_id: int
    condition: str
    genre: str
    turns_executed: int
    response_mean_s: float
    response_p50_s: float
    response_p90_s: float
    response_p95_s: float
    response_std_s: float
    response_min_s: float
    response_max_s: float
    auto_metrics: dict[str, float]
    llm_judge: dict[str, int | float]
    error: str = ""


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * p
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_vals[low]
    frac = pos - low
    return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * frac


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": statistics.fmean(latencies),
        "p50": _percentile(latencies, 0.50),
        "p90": _percentile(latencies, 0.90),
        "p95": _percentile(latencies, 0.95),
        "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
    }


def _default_auto_metrics() -> dict[str, float]:
    return {k: 0.0 for k in AUTO_KEYS}


def _default_llm_scores() -> dict[str, int | float]:
    data: dict[str, int | float] = {d: 0 for d in DIMENSIONS}
    data["average"] = 0.0
    return data


def _disable_kg_runtime(engine: Any) -> None:
    """Disable KG update/summary/conflict branches for a single engine instance."""

    def _noop_apply_kg_update(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    def _empty_summary(self: Any) -> str:
        return ""

    def _no_conflicts(new_text: str = "") -> list[dict[str, str]]:
        return []

    engine._apply_kg_update = types.MethodType(_noop_apply_kg_update, engine)
    engine._current_kg_summary = types.MethodType(_empty_summary, engine)
    engine.conflict_det.check_all = _no_conflicts


def run_one_session(session_id: int, condition: str, genre: str, max_turns: int) -> SessionResult:
    from src.engine.game_engine import GameEngine
    from src.evaluation.llm_judge import judge
    from src.evaluation.metrics import full_evaluation

    latencies: list[float] = []
    kg_enabled = condition == "kg_on"

    try:
        engine = GameEngine(genre=genre)
        if not kg_enabled:
            _disable_kg_runtime(engine)

        opening = engine.start_game()
        if not kg_enabled:
            _disable_kg_runtime(engine)

        options = opening.options
        turns = 0

        while turns < max_turns and options:
            action = options[0].text
            t0 = time.perf_counter()
            result = engine.process_turn(action)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            options = result.options
            turns += 1

        auto = full_evaluation(
            texts=engine.all_story_texts,
            entity_names=engine.kg_entity_names,
            turn_conflict_counts=engine.turn_conflict_counts,
            kg_turn_stats=engine.kg_density_inputs,
        )
        llm_scores = judge("\n".join(engine.all_story_texts))
        stats = _latency_stats(latencies)

        return SessionResult(
            session_id=session_id,
            condition=condition,
            genre=genre,
            turns_executed=turns,
            response_mean_s=stats["mean"],
            response_p50_s=stats["p50"],
            response_p90_s=stats["p90"],
            response_p95_s=stats["p95"],
            response_std_s=stats["std"],
            response_min_s=stats["min"],
            response_max_s=stats["max"],
            auto_metrics={k: float(auto.get(k, 0.0)) for k in AUTO_KEYS},
            llm_judge={
                **{d: int(llm_scores.get(d, 0)) for d in DIMENSIONS},
                "average": float(llm_scores.get("average", 0.0)),
            },
        )
    except Exception as exc:
        stats = _latency_stats(latencies)
        return SessionResult(
            session_id=session_id,
            condition=condition,
            genre=genre,
            turns_executed=len(latencies),
            response_mean_s=stats["mean"],
            response_p50_s=stats["p50"],
            response_p90_s=stats["p90"],
            response_p95_s=stats["p95"],
            response_std_s=stats["std"],
            response_min_s=stats["min"],
            response_max_s=stats["max"],
            auto_metrics=_default_auto_metrics(),
            llm_judge=_default_llm_scores(),
            error=str(exc),
        )


def _aggregate_condition(results: list[SessionResult], condition: str) -> dict[str, Any]:
    rows = [r for r in results if r.condition == condition]
    if not rows:
        return {}

    means = [r.response_mean_s for r in rows]
    out: dict[str, Any] = {
        "condition": condition,
        "sessions": len(rows),
        "turns_total": sum(r.turns_executed for r in rows),
        "response_mean_s": statistics.fmean(means),
        "response_p50_s": _percentile(means, 0.50),
        "response_p90_s": _percentile(means, 0.90),
        "response_p95_s": _percentile(means, 0.95),
        "response_std_s": statistics.stdev(means) if len(means) > 1 else 0.0,
        "response_min_s": min(means),
        "response_max_s": max(means),
        "auto_metrics": {},
        "llm_judge": {},
        "error_sessions": sum(1 for r in rows if r.error),
    }

    for key in AUTO_KEYS:
        vals = [r.auto_metrics.get(key, 0.0) for r in rows]
        out["auto_metrics"][key] = statistics.fmean(vals)

    for dim in [*DIMENSIONS, "average"]:
        vals = [float(r.llm_judge.get(dim, 0.0)) for r in rows]
        out["llm_judge"][dim] = statistics.fmean(vals)

    return out


def _delta_map(on_data: dict[str, float], off_data: dict[str, float]) -> dict[str, float]:
    keys = set(on_data.keys()) | set(off_data.keys())
    return {k: float(on_data.get(k, 0.0)) - float(off_data.get(k, 0.0)) for k in sorted(keys)}


def _fmt4(value: float) -> str:
    return f"{value:.4f}"


def _fmt2(value: float) -> str:
    return f"{value:.2f}"


def _render_report(
    *,
    genre: str,
    turns: int,
    runs: int,
    results: list[SessionResult],
    kg_on: dict[str, Any],
    kg_off: dict[str, Any],
    latency_delta: dict[str, float],
    auto_delta: dict[str, float],
    llm_delta: dict[str, float],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# KG 开关工况对比报告")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- Genre: {genre}")
    lines.append(f"- 每工况运行次数: {runs}")
    lines.append(f"- 每次最大轮次: {turns}")
    lines.append("- 动作策略: 每回合固定选择第一个选项")
    lines.append("")

    lines.append("## 1. 会话明细")
    lines.append("")
    lines.append("| Session | Condition | Genre | Turns | Mean(s) | P50(s) | P90(s) | P95(s) | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in results:
        status = "OK" if not row.error else f"ERROR: {row.error[:40]}"
        lines.append(
            "| "
            f"{row.session_id} | {row.condition} | {row.genre} | {row.turns_executed} | "
            f"{_fmt4(row.response_mean_s)} | {_fmt4(row.response_p50_s)} | {_fmt4(row.response_p90_s)} | "
            f"{_fmt4(row.response_p95_s)} | {status} |"
        )
    lines.append("")

    lines.append("## 2. 延迟指标对比（kg_on - kg_off）")
    lines.append("")
    lines.append("| Metric | kg_on | kg_off | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in [
        "response_mean_s",
        "response_p50_s",
        "response_p90_s",
        "response_p95_s",
        "response_std_s",
        "response_min_s",
        "response_max_s",
    ]:
        lines.append(
            f"| {key} | {_fmt4(float(kg_on.get(key, 0.0)))} | {_fmt4(float(kg_off.get(key, 0.0)))} | "
            f"{_fmt4(float(latency_delta.get(key, 0.0)))} |"
        )
    lines.append("")

    lines.append("## 3. 自动指标对比（metrics.py）")
    lines.append("")
    lines.append("| Metric | kg_on | kg_off | Delta |")
    lines.append("|---|---:|---:|---:|")
    on_auto = kg_on.get("auto_metrics", {})
    off_auto = kg_off.get("auto_metrics", {})
    for key in AUTO_KEYS:
        lines.append(
            f"| {key} | {_fmt4(float(on_auto.get(key, 0.0)))} | {_fmt4(float(off_auto.get(key, 0.0)))} | "
            f"{_fmt4(float(auto_delta.get(key, 0.0)))} |"
        )
    lines.append("")

    lines.append("## 4. LLM Judge 对比（llm_judge.py）")
    lines.append("")
    lines.append("| Dimension | kg_on | kg_off | Delta |")
    lines.append("|---|---:|---:|---:|")
    on_llm = kg_on.get("llm_judge", {})
    off_llm = kg_off.get("llm_judge", {})
    for key in [*DIMENSIONS, "average"]:
        lines.append(
            f"| {key} | {_fmt2(float(on_llm.get(key, 0.0)))} | {_fmt2(float(off_llm.get(key, 0.0)))} | "
            f"{_fmt2(float(llm_delta.get(key, 0.0)))} |"
        )
    lines.append("")

    lines.append("## 5. 说明")
    lines.append("")
    lines.append("- `Delta` 统一定义为 `kg_on - kg_off`。")
    lines.append("- 本次为单次对比（runs=1），结果会受模型随机性与 API 抖动影响。")
    lines.append("- 若评测模型不可用，LLM Judge 维度会回退到 0。")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG on/off benchmark for StoryWeaver")
    parser.add_argument("--genre", type=str, default="fantasy", help="Genre for both conditions")
    parser.add_argument("--turns", type=int, default=5, help="Max turns per session")
    parser.add_argument("--runs", type=int, default=1, help="Runs per condition")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/kg_on_off_report.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="docs/kg_on_off_results.json",
        help="Raw JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    genre = str(args.genre).strip() or "fantasy"
    turns = max(1, int(args.turns))
    runs = max(1, int(args.runs))

    conditions = ["kg_on", "kg_off"]
    session_id = 1
    results: list[SessionResult] = []

    for condition in conditions:
        for _ in range(runs):
            result = run_one_session(
                session_id=session_id,
                condition=condition,
                genre=genre,
                max_turns=turns,
            )
            results.append(result)
            session_id += 1

    agg_on = _aggregate_condition(results, "kg_on")
    agg_off = _aggregate_condition(results, "kg_off")

    latency_delta = _delta_map(
        {k: float(agg_on.get(k, 0.0)) for k in [
            "response_mean_s",
            "response_p50_s",
            "response_p90_s",
            "response_p95_s",
            "response_std_s",
            "response_min_s",
            "response_max_s",
        ]},
        {k: float(agg_off.get(k, 0.0)) for k in [
            "response_mean_s",
            "response_p50_s",
            "response_p90_s",
            "response_p95_s",
            "response_std_s",
            "response_min_s",
            "response_max_s",
        ]},
    )
    auto_delta = _delta_map(
        dict(agg_on.get("auto_metrics", {})),
        dict(agg_off.get("auto_metrics", {})),
    )
    llm_delta = _delta_map(
        dict(agg_on.get("llm_judge", {})),
        dict(agg_off.get("llm_judge", {})),
    )

    report_text = _render_report(
        genre=genre,
        turns=turns,
        runs=runs,
        results=results,
        kg_on=agg_on,
        kg_off=agg_off,
        latency_delta=latency_delta,
        auto_delta=auto_delta,
        llm_delta=llm_delta,
    )

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")

    raw_path = PROJECT_ROOT / args.raw_output
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "genre": genre,
            "turns": turns,
            "runs_per_condition": runs,
            "conditions": conditions,
            "policy": "always_pick_first_option",
        },
        "sessions": [asdict(r) for r in results],
        "aggregate": {
            "kg_on": agg_on,
            "kg_off": agg_off,
            "delta": {
                "latency": latency_delta,
                "auto_metrics": auto_delta,
                "llm_judge": llm_delta,
            },
        },
    }
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written to: {out_path}")
    print(f"Raw results written to: {raw_path}")


if __name__ == "__main__":
    main()
