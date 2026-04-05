"""Run automated StoryWeaver evaluation benchmark and write a markdown report.

Default plan in this script:
- 3 sessions total: fantasy, sci-fi, mystery (one each)
- 0 warmup sessions
- up to 10 turns per session
- deterministic interaction policy: always pick the first branch option

Collected outputs:
- automatic metrics from ``src.evaluation.metrics.full_evaluation``
- LLM-judge metrics from ``src.evaluation.llm_judge.judge``
- response-time stats from ``GameEngine.process_turn`` latency
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
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


def run_one_session(session_id: int, genre: str, max_turns: int) -> SessionResult:
    from src.engine.game_engine import GameEngine
    from src.evaluation.llm_judge import judge
    from src.evaluation.metrics import full_evaluation

    latencies: list[float] = []
    try:
        engine = GameEngine(genre=genre)
        opening = engine.start_game()
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
    except Exception as exc:  # pragma: no cover - runtime safety branch
        stats = _latency_stats(latencies)
        return SessionResult(
            session_id=session_id,
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


def _aggregate(results: list[SessionResult]) -> dict[str, Any]:
    if not results:
        return {}

    all_turn_lat_means = [r.response_mean_s for r in results]
    agg: dict[str, Any] = {
        "sessions": len(results),
        "turns_total": sum(r.turns_executed for r in results),
        "response_mean_s": statistics.fmean(all_turn_lat_means),
        "response_p50_s": _percentile(all_turn_lat_means, 0.50),
        "response_p90_s": _percentile(all_turn_lat_means, 0.90),
        "response_p95_s": _percentile(all_turn_lat_means, 0.95),
        "response_std_s": statistics.stdev(all_turn_lat_means) if len(all_turn_lat_means) > 1 else 0.0,
        "response_min_s": min(all_turn_lat_means),
        "response_max_s": max(all_turn_lat_means),
        "auto_metrics": {},
        "llm_judge": {},
        "error_sessions": sum(1 for r in results if r.error),
    }

    for key in AUTO_KEYS:
        vals = [r.auto_metrics.get(key, 0.0) for r in results]
        agg["auto_metrics"][key] = statistics.fmean(vals)

    for dim in [*DIMENSIONS, "average"]:
        vals = [float(r.llm_judge.get(dim, 0.0)) for r in results]
        agg["llm_judge"][dim] = statistics.fmean(vals)

    return agg


def _fmt4(value: float) -> str:
    return f"{value:.4f}"


def _fmt2(value: float) -> str:
    return f"{value:.2f}"


def _render_report(
    *,
    genres: list[str],
    max_turns: int,
    warmup: int,
    results: list[SessionResult],
    aggregate: dict[str, Any],
) -> str:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# StoryWeaver 自动化测试报告")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append("- 测试目标: 评估系统自动指标、LLM 主观评分与端到端响应时间")
    lines.append("")

    lines.append("## 1. 测试配置")
    lines.append("")
    lines.append(f"- 会话总数: {len(results)}")
    lines.append(f"- Genre 列表: {', '.join(genres)}")
    lines.append(f"- 每会话最大轮次 (max_turns): {max_turns}")
    lines.append(f"- 预热轮次 (warmup): {warmup}")
    lines.append("- 动作策略: 每回合自动选择 branch options 的第一个选项")
    lines.append("")

    lines.append("## 2. 会话明细")
    lines.append("")
    lines.append("| Session | Genre | Turns | Mean(s) | P50(s) | P90(s) | P95(s) | Std(s) | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        status = "OK" if not r.error else f"ERROR: {r.error[:40]}"
        lines.append(
            "| "
            f"{r.session_id} | {r.genre} | {r.turns_executed} | {_fmt4(r.response_mean_s)} | "
            f"{_fmt4(r.response_p50_s)} | {_fmt4(r.response_p90_s)} | {_fmt4(r.response_p95_s)} | "
            f"{_fmt4(r.response_std_s)} | {status} |"
        )
    lines.append("")

    lines.append("## 3. 总体汇总（不按 Genre 分组）")
    lines.append("")
    lines.append("### 3.1 响应时间")
    lines.append("")
    lines.append("| Metric | Value (s) |")
    lines.append("|---|---:|")
    lines.append(f"| Mean | {_fmt4(float(aggregate.get('response_mean_s', 0.0)))} |")
    lines.append(f"| P50 | {_fmt4(float(aggregate.get('response_p50_s', 0.0)))} |")
    lines.append(f"| P90 | {_fmt4(float(aggregate.get('response_p90_s', 0.0)))} |")
    lines.append(f"| P95 | {_fmt4(float(aggregate.get('response_p95_s', 0.0)))} |")
    lines.append(f"| Std | {_fmt4(float(aggregate.get('response_std_s', 0.0)))} |")
    lines.append(f"| Min | {_fmt4(float(aggregate.get('response_min_s', 0.0)))} |")
    lines.append(f"| Max | {_fmt4(float(aggregate.get('response_max_s', 0.0)))} |")
    lines.append("")

    lines.append("### 3.2 自动指标（metrics.py）")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    auto_data = aggregate.get("auto_metrics", {})
    for key in AUTO_KEYS:
        lines.append(f"| {key} | {_fmt4(float(auto_data.get(key, 0.0)))} |")
    lines.append("")

    lines.append("### 3.3 LLM Judge（llm_judge.py）")
    lines.append("")
    lines.append("| Dimension | Value |")
    lines.append("|---|---:|")
    llm_data = aggregate.get("llm_judge", {})
    for key in [*DIMENSIONS, "average"]:
        lines.append(f"| {key} | {_fmt2(float(llm_data.get(key, 0.0)))} |")
    lines.append("")

    lines.append("## 4. 参数解释与解读")
    lines.append("")
    lines.append("- `max_turns=10`: 每会话最多 10 次用户动作，控制测试长度与成本。")
    lines.append("- `warmup=0`: 不做预热，报告结果会包含冷启动影响。")
    lines.append("- 动作策略固定首选项: 提高实验可复现性，减少随机动作导致的方差。")
    lines.append("- `distinct_n`: 越高通常代表词汇/短语多样性更好。")
    lines.append("- `self_bleu`: 越低通常代表重复度更低、生成更不模板化。")
    lines.append("- `entity_coverage`: 越高表示生成文本覆盖更多知识图谱实体。")
    lines.append("- `consistency_rate`: 越高表示每回合冲突更少、世界状态更稳定。")
    lines.append("- `type_token_ratio`: 词汇丰富度（不同词占比），越高通常表示词汇变化更丰富。")
    lines.append("- `flesch_reading_ease`: 可读性分数（越高越易读）。")
    lines.append("- `lexical_overlap`: 相邻叙事文本词汇交集比例，用于衡量局部承接。")
    lines.append("- `graph_density_average`: 会话期内 KG 平均密度，用于观察世界关系复杂度。")
    lines.append("- `graph_density_delta`: 会话结束与开场的 KG 密度变化量。")
    lines.append("- LLM Judge 八维评分（1-10）用于补充主观质量评估，`average` 为八维均值。")
    lines.append("- 新增三维含义：`option_relevance`（选项相关性）、`causal_link`（因果链合理性）、`local_coherence`（相邻回合连贯性）。")
    lines.append("")

    lines.append("## 5. 备注")
    lines.append("")
    lines.append(
        f"- 错误会话数: {int(aggregate.get('error_sessions', 0))}。"
        "若 API 或模型加载失败，对应会话将记为 ERROR，并以 0 值填充指标。"
    )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StoryWeaver automated evaluation benchmark")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per session")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup sessions (excluded from report)")
    parser.add_argument(
        "--nlg-mode",
        type=str,
        default=None,
        choices=["local", "api", "hybrid"],
        help="NLG mode to use (overrides .env NLG_MODE)",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=["fantasy", "sci-fi", "mystery"],
        help="Genres to test, one session per genre",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/automated_test_report.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="docs/automated_test_results.json",
        help="Raw JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.nlg_mode:
        os.environ["NLG_MODE"] = args.nlg_mode

    warmup = max(0, int(args.warmup))
    max_turns = max(1, int(args.max_turns))
    genres = [g.strip() for g in args.genres if g.strip()]
    if not genres:
        raise ValueError("At least one genre is required")

    all_results: list[SessionResult] = []
    session_id = 1

    for _ in range(warmup):
        _ = run_one_session(session_id=session_id, genre=genres[0], max_turns=max_turns)
        session_id += 1

    for genre in genres:
        result = run_one_session(session_id=session_id, genre=genre, max_turns=max_turns)
        all_results.append(result)
        session_id += 1

    aggregate = _aggregate(all_results)
    report_text = _render_report(
        genres=genres,
        max_turns=max_turns,
        warmup=warmup,
        results=all_results,
        aggregate=aggregate,
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    raw_path = PROJECT_ROOT / args.raw_output
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "max_turns": max_turns,
            "warmup": warmup,
            "genres": genres,
            "policy": "always_pick_first_option",
        },
        "sessions": [asdict(r) for r in all_results],
        "aggregate": aggregate,
    }
    raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written to: {output_path}")
    print(f"Raw results written to: {raw_path}")


if __name__ == "__main__":
    main()
