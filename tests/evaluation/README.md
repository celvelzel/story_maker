# Quality Benchmark & Regression Harness

This directory contains the Wave-A quality benchmark foundation for NLU + KG quality-first development.

## Files

- `data/nlu_kg_quality_benchmark.jsonl`
  - Golden dataset (>=120 cases) with six mandatory scenario groups:
    - `pronoun_heavy_dialogue`
    - `alias_heavy_entities`
    - `conflicting_facts`
    - `temporal_updates`
    - `ambiguous_intent`
    - `long_context_continuation`
- `test_quality_benchmark_schema.py`
  - Deterministic schema/shape checks for the JSONL benchmark.
- `quality_runner.py`
  - CLI runner for baseline/compare reports.
- `test_quality_regression.py`
  - Deterministic regression harness tests.
- `test_quality_gates.py`
  - Executable policy checks for Gate-1/2/3.

## Benchmark Case Contract

Each JSONL line is one object with these required keys:

- `id: str` (unique)
- `scenario_tag: str` (one of the six tags above)
- `acceptance_tags: list[str]` (non-empty)
- `input: str` (non-empty)
- `prior_context: list[str]` (non-empty)
- `expected_entities: list[object]`
  - each entity object must have at least `name: str`, `type: str`
- `expected_relations: list[object]`
  - each relation object must have at least `source: str`, `target: str`, `relation: str`
- `expected_conflicts: list[object]`
  - each conflict object must have at least `type: str`, `entity: str`
- `expected_intent: str`

Intent labels currently expected by schema test:

- `explore`, `action`, `dialogue`, `use_item`, `ask_info`, `rest`, `trade`

Entity types currently expected by schema test:

- `person`, `location`, `item`, `creature`, `event`

## Commands

Validate benchmark schema:

```bash
pytest tests/evaluation/test_quality_benchmark_schema.py -q
```

Run integration safety check after benchmark updates:

```bash
pytest tests/integration/test_integration.py -q
```

Generate baseline metrics report:

```bash
python -m tests.evaluation.quality_runner --mode baseline
```

Compare against baseline:

```bash
python -m tests.evaluation.quality_runner --mode compare --against baseline
```

Run regression tests:

```bash
pytest tests/evaluation/test_quality_regression.py -q
pytest tests/evaluation/test_quality_gates.py -q
```

## Quality Gate Policy (Executable)

- Gate-1 (must pass): no tracked metric may regress by more than 1.0 percentage point versus baseline.
- Gate-2 (target):
  - `relation_f1` gain >= +3pp
  - `coreference_accuracy` gain >= +5pp
  - `contradiction_rate` <= 80% of baseline (20% relative reduction)
- Gate-3 (latency-phase entry): Gate-1 + Gate-2 must pass in two consecutive compare runs.

The gate logic is implemented in `tests/evaluation/quality_runner.py::evaluate_quality_gates` and enforced by `tests/evaluation/test_quality_gates.py`.

## Policy Notes

- This benchmark is deterministic input/output scaffolding for regression gating.
- Latency optimization work must not start until quality gates are defined and passing.
