# NLU + Knowledge Graph Quality-First Optimization Roadmap

## Objective
- Produce a decision-complete execution roadmap that prioritizes **generation quality and accuracy** first, with **latency optimization second**.
- Keep all actions bounded to NLU/KG-related modules and their direct orchestration/evaluation paths.

## Confirmed Priorities
1. Primary: generation effect quality and accuracy.
2. Secondary: latency.

## Scope

### In Scope
- NLU quality and robustness improvements in:
  - `src/nlu/intent_classifier.py`
  - `src/nlu/entity_extractor.py`
  - `src/nlu/coreference.py`
  - `src/nlu/sentiment_analyzer.py`
- KG extraction/consistency quality improvements in:
  - `src/knowledge_graph/relation_extractor.py`
  - `src/knowledge_graph/conflict_detector.py`
  - `src/knowledge_graph/graph.py`
- Orchestration and observability adjustments in:
  - `src/engine/game_engine.py`
  - `src/utils/api_client.py`
  - `src/evaluation/metrics.py`
  - `src/evaluation/llm_judge.py`
  - `app.py` (only for metric exposure / diagnostics)
- Test and benchmark expansion in:
  - `tests/nlu/*`
  - `tests/kg/*`
  - `tests/integration/*`
  - new evaluation/performance tests under `tests/`

### Out of Scope
- Full product redesign, UI redesign, or migration away from Streamlit/NetworkX.
- New external storage backend (e.g., graph DB) in this roadmap.
- Large model replacement program (e.g., wholesale migration to different model families).

## Constraints and Guardrails
- Quality-first gate: no latency optimization may be merged unless quality gates pass with no regression.
- Every optimization task must include explicit QA scenarios with deterministic assertions where possible.
- Any heuristic threshold change must include before/after evaluation artifacts.
- Keep feature-flag/strategy fallback paths for risky behavior changes.

## Baseline Evidence Snapshot (from exploration)
- Pipeline order confirmed: `Coref -> Intent -> Sentiment -> Entity -> Story -> KG update -> Conflict -> Options` in `src/engine/game_engine.py`.
- Existing KG/NLU test infrastructure is present across `tests/nlu`, `tests/kg`, `tests/integration`, `tests/engine`.
- Existing telemetry exists but is coarse-grained:
  - Turn-level elapsed time in `app.py`
  - Token/cost tracking in `src/utils/api_client.py`
- Known hotspots identified:
  - Quality: relation extraction reliability, conflict-resolution quality, coreference fallback edge cases, entity fuzzy-match precision.
  - Performance: repeated summary generation, full-graph recalculation/decay each turn.

## Success Criteria

### Quality/Accuracy (Primary)
- KG consistency and contradiction handling quality improves on benchmark set with no regressions in existing test suites.
- Entity extraction correctness improves on curated cases (including aliasing, pronouns, and ambiguous entities).
- Coreference correctness improves for multi-pronoun turns and entity-type-constrained contexts.
- Story-context fidelity (entities/relations reflected in generated narrative options) improves under evaluation runs.

### Latency (Secondary)
- Achieve measurable turn-time reduction only after quality gates pass.
- Maintain identical or improved quality metrics after latency changes.

## Execution Strategy (Phased)
- Phase 0: evaluation foundation and quality gates.
- Phase 1: quality-critical NLU and KG improvements.
- Phase 2: quality-protected latency optimization.
- Phase 3: stabilization, release guardrails, and final verification.

## Task Dependency Graph
- Phase 0 tasks block all downstream phases.
- Phase 1 quality tasks must complete before Phase 2 latency tasks.
- Phase 2 tasks can run partially in parallel after required upstream dependencies are met.
- Final verification depends on all tasks.

## Parallelization Plan
- Wave A (parallel): baseline instrumentation + benchmark dataset construction.
- Wave B (parallel): coreference/entity-extraction quality track + KG extraction/conflict quality track.
- Wave C (parallel): safe latency optimizations (summary caching, incremental recalculation) behind quality gates.
- Wave D (sequential): full regression + acceptance verification + rollout playbook.

## Task List

### Wave A — Quality Baseline Foundation (must complete first)

#### Task A1 — Build quality benchmark corpus and scoring spec
- **Purpose**: Create a stable benchmark so all quality changes are measured against the same cases.
- **Files to create/update**:
  - `tests/evaluation/data/nlu_kg_quality_benchmark.jsonl` (new)
  - `tests/evaluation/test_quality_benchmark_schema.py` (new)
  - `tests/evaluation/README.md` (new)
- **Implementation decisions (fixed)**:
  - Benchmark must include minimum 120 turns across 6 scenario groups: pronoun-heavy dialogue, alias-heavy entities, conflicting facts, temporal updates, ambiguous intent, long-context continuation.
  - Each case must store: input, prior context, expected entities, expected relations/conflicts, expected intent (if applicable), and acceptance tags.
  - Schema validation via deterministic test (required fields + type checks + non-empty constraints).
- **Dependencies**: none.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/evaluation/test_quality_benchmark_schema.py -q`
     - Assert: 100% pass, no missing mandatory fields.
  2. Run: `pytest tests/integration/test_integration.py -q`
     - Assert: existing integration tests still pass after benchmark artifact introduction.
- **Acceptance criteria**:
  - Benchmark file exists with >=120 valid cases and scenario tags.
  - Schema test passes in CI.

#### Task A2 — Implement automated quality evaluation harness
- **Purpose**: Convert benchmark corpus into reproducible accuracy/quality metrics.
- **Files to create/update**:
  - `tests/evaluation/test_quality_regression.py` (new)
  - `tests/evaluation/quality_runner.py` (new)
  - `src/evaluation/metrics.py` (extend metric helpers if needed)
- **Implementation decisions (fixed)**:
  - Harness computes and stores: entity precision/recall/F1, relation precision/recall/F1, contradiction rate, coreference resolution accuracy, intent accuracy, consistency_rate.
  - Output must be machine-readable JSON in `tests/evaluation/reports/latest_quality.json`.
  - Baseline snapshot command is fixed: `python -m tests.evaluation.quality_runner --mode baseline`.
- **Dependencies**: Task A1.
- **QA scenarios (mandatory)**:
  1. Run: `python -m tests.evaluation.quality_runner --mode baseline`
     - Assert: report JSON generated with all required metric keys.
  2. Run: `pytest tests/evaluation/test_quality_regression.py -q`
     - Assert: parser/scorer deterministic on fixed fixture subset.
- **Acceptance criteria**:
  - Baseline report committed as reference artifact.
  - Harness reproducibility confirmed by two consecutive identical runs on fixed seed.

#### Task A3 — Add stage-level observability for quality debugging (not optimization yet)
- **Purpose**: Enable root-cause analysis when quality scores regress.
- **Files to create/update**:
  - `src/engine/game_engine.py`
  - `src/utils/api_client.py`
  - `app.py` (diagnostic-only display)
- **Implementation decisions (fixed)**:
  - Record per-turn stage timings for: coref, intent, sentiment, entity extraction, story generation, KG extraction/update, conflict, options.
  - Record per-stage LLM token/cost attribution where applicable.
  - Add `nlu_debug.stage_metrics` payload; no behavior change to story logic.
- **Dependencies**: Task A2.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/engine/test_engine_enhanced.py -q`
     - Assert: engine output remains backward-compatible plus optional stage metrics.
  2. Run: `pytest tests/integration/test_integration.py -q`
     - Assert: no pipeline regressions from telemetry additions.
- **Acceptance criteria**:
  - Stage metrics are present in debug outputs for each turn.
  - No functional diff in generated outputs under fixed mocks.

#### Task A4 — Establish quality gate thresholds (default policy)
- **Purpose**: Freeze objective pass/fail rules before quality refactors.
- **Files to create/update**:
  - `tests/evaluation/test_quality_gates.py` (new)
  - `tests/evaluation/README.md`
- **Implementation decisions (fixed defaults)**:
  - Gate-1 (must pass): no metric may regress >1.0 percentage point from baseline.
  - Gate-2 (target): relation F1 +3pp, coreference accuracy +5pp, contradiction rate -20% relative.
  - Gate-3 (latency phase entry): quality gates must be green for two consecutive runs.
- **Dependencies**: Tasks A1–A3.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/evaluation/test_quality_gates.py -q`
     - Assert: gate logic rejects synthetic regression fixtures.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: gate summary produced with PASS/FAIL statuses.
- **Acceptance criteria**:
  - Quality gate policy encoded as executable tests (not prose-only).

### Wave B — Quality-Critical Improvements (primary value wave)

#### Task B1 — Fix coreference fallback multi-pronoun handling and type constraints
- **Purpose**: Improve pronoun resolution correctness in rule fallback path, especially in long or ambiguous turns.
- **Files to create/update**:
  - `src/nlu/coreference.py`
  - `tests/nlu/test_coreference_enhanced.py`
  - `tests/evaluation/test_quality_regression.py` (add pronoun-heavy benchmark assertions)
- **Implementation decisions (fixed)**:
  - Remove single-replacement early-exit behavior in fallback loops; process all eligible pronoun matches per sentence.
  - Preserve entity-type-aware mapping (person/location/item/etc.) and add explicit conflict tie-break rules.
  - Add protection against over-replacement in quoted dialogue segments.
- **Dependencies**: Wave A complete.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/nlu/test_coreference_enhanced.py -q`
     - Assert: multi-pronoun fixtures resolve all expected pronouns.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: coreference accuracy improves >=5pp and no gate regression.
- **Acceptance criteria**:
  - Coreference benchmark target met.
  - No regressions in existing NLU integration tests.

#### Task B2 — Improve entity extraction precision with alias normalization and confidence calibration
- **Purpose**: Reduce false positives/false merges while preserving recall in KG-linked entity extraction.
- **Files to create/update**:
  - `src/nlu/entity_extractor.py`
  - `tests/nlu/test_entity_extractor_enhanced.py`
  - `tests/evaluation/data/nlu_kg_quality_benchmark.jsonl` (alias/ambiguity cases)
- **Implementation decisions (fixed)**:
  - Introduce deterministic alias normalization map layer before fuzzy matching.
  - Split fuzzy thresholds by entity type (person/location/item) with fixed initial defaults.
  - Add confidence output per extracted entity and block low-confidence auto-upserts to KG unless corroborated by context.
- **Dependencies**: Wave A complete.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/nlu/test_entity_extractor_enhanced.py -q`
     - Assert: alias fixtures merge correctly; ambiguous fixtures avoid incorrect merges.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: entity precision improves with <=1pp recall drop (or recall non-regression preferred).
- **Acceptance criteria**:
  - Entity extraction quality improves under gate policy.

#### Task B3 — Harden KG relation extraction schema and contradiction sensitivity
- **Purpose**: Increase factual correctness of extracted relations and reduce malformed extraction payloads.
- **Files to create/update**:
  - `src/knowledge_graph/relation_extractor.py`
  - `tests/kg/test_relation_extractor_enhanced.py`
  - `tests/kg/test_kg_type_validation.py`
- **Implementation decisions (fixed)**:
  - Enforce strict schema normalization for entity/relation types and confidence bounds.
  - Add invalid-relation quarantine path (discard + debug event) instead of permissive acceptance.
  - Expand extraction prompt constraints toward canonical relation set and temporal clarity.
- **Dependencies**: Wave A complete.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/kg/test_relation_extractor_enhanced.py tests/kg/test_kg_type_validation.py -q`
     - Assert: malformed outputs are rejected/quarantined deterministically.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: relation F1 improves >=3pp; contradiction rate non-increasing.
- **Acceptance criteria**:
  - Relation quality target met with no schema-acceptance regressions.

#### Task B4 — Upgrade conflict detection policy with quality-safe arbitration defaults
- **Purpose**: Improve contradiction handling quality while avoiding hallucinated conflict decisions.
- **Files to create/update**:
  - `src/knowledge_graph/conflict_detector.py`
  - `tests/kg/test_conflict_resolution.py`
  - `tests/kg/test_temporal_reasoning.py`
- **Implementation decisions (fixed)**:
  - Keep deterministic rule/temporal checks as first-class source of truth.
  - LLM arbitration remains optional and only triggered on unresolved high-impact conflicts.
  - Add confidence band policy: low-confidence conflicts go to deferred observation queue instead of immediate rewrite.
- **Dependencies**: Task B3.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/kg/test_conflict_resolution.py tests/kg/test_temporal_reasoning.py -q`
     - Assert: deterministic fixtures yield expected conflict outcomes.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: contradiction rate decreases by >=20% relative from baseline.
- **Acceptance criteria**:
  - Conflict quality gate passes; no new false contradiction spikes.

### Wave C — Latency Optimization Under Quality Lock (secondary)

#### Task C1 — Cache per-turn KG summary and eliminate duplicate traversals
- **Purpose**: Reduce redundant CPU work without altering semantic outputs.
- **Files to create/update**:
  - `src/engine/game_engine.py`
  - `src/knowledge_graph/graph.py` (only if helper needed)
  - `tests/engine/test_engine_enhanced.py`
  - `tests/performance/test_turn_latency.py` (new)
- **Implementation decisions (fixed)**:
  - Generate KG summary once per turn after KG update and reuse in downstream consumers.
  - Cache scope is strictly per-turn (no cross-turn caching) to avoid stale state risk.
  - Any cache invalidation complexity beyond per-turn is explicitly out of scope.
- **Dependencies**: Wave B complete and quality gates green.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/engine/test_engine_enhanced.py -q`
     - Assert: story/options/conflict inputs remain semantically identical on fixed fixtures.
  2. Run: `pytest tests/performance/test_turn_latency.py -q`
     - Assert: median turn latency improves vs Wave B baseline.
  3. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: all quality gates remain PASS.
- **Acceptance criteria**:
  - Duplicate summary calls reduced to one per turn path.
  - No quality regression.

#### Task C2 — Introduce incremental KG importance recalculation with equivalence checks
- **Purpose**: Cut full-graph recomputation overhead while preserving KG scoring behavior.
- **Files to create/update**:
  - `src/knowledge_graph/graph.py`
  - `tests/kg/test_graph_enhanced.py`
  - `tests/performance/test_kg_recalc_latency.py` (new)
- **Implementation decisions (fixed)**:
  - Recalculate importance only for touched nodes + neighborhood as default strategy.
  - Keep full recomputation as fallback strategy flag for safety (`KG_IMPORTANCE_MODE`).
  - Add deterministic equivalence test comparing incremental vs full on fixed update sequences.
- **Dependencies**: Task C1.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/kg/test_graph_enhanced.py -q`
     - Assert: incremental and full modes produce equivalent rankings on fixtures.
  2. Run: `pytest tests/performance/test_kg_recalc_latency.py -q`
     - Assert: recalculation wall-time improvement with growing node counts.
  3. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: quality gate remains PASS.
- **Acceptance criteria**:
  - Incremental mode becomes default only after equivalence assertions pass.

#### Task C3 — Tune decay/update cadence with quality-preserving policy
- **Purpose**: Reduce unnecessary per-turn decay cost while preserving narrative consistency behavior.
- **Files to create/update**:
  - `src/knowledge_graph/graph.py`
  - `config.py`
  - `tests/kg/test_temporal_reasoning.py`
  - `tests/performance/test_kg_decay_cost.py` (new)
- **Implementation decisions (fixed)**:
  - Introduce configurable decay cadence (e.g., every N turns) with default conservative value.
  - Enforce immediate decay pass on high-conflict turns regardless of cadence.
  - All cadence changes must be guarded by feature flag and regression-tested.
- **Dependencies**: Task C2.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/kg/test_temporal_reasoning.py -q`
     - Assert: temporal correctness remains valid under new cadence.
  2. Run: `pytest tests/performance/test_kg_decay_cost.py -q`
     - Assert: measurable decay-phase cost reduction.
  3. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline`
     - Assert: no quality gate violations.
- **Acceptance criteria**:
  - Decay cadence optimization accepted only if all quality gates pass.

### Wave D — Stabilization, Documentation, and Rollout

#### Task D1 — Consolidate strategy flags and defaults for safe rollout
- **Purpose**: Ensure deploy-time behavior is explicit and reversible.
- **Files to create/update**:
  - `config.py`
  - `docs/guides/technical-route.md`
  - `docs/reports/nlu-kg-improvement.md` (append rollout section)
- **Implementation decisions (fixed)**:
  - Document final default values for extraction/conflict/importance/decay strategies.
  - Include rollback toggles for each high-risk feature.
  - Add compatibility note for fastcoref/transformers patch expectations.
- **Dependencies**: Waves B and C complete.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/engine/test_engine_enhanced.py -q`
     - Assert: configuration permutations load correctly.
  2. Run: `python scripts/health_check.py`
     - Assert: startup checks pass with final defaults.
- **Acceptance criteria**:
  - Rollout and rollback instructions complete and test-validated.

#### Task D2 — Final regression matrix and release evidence pack
- **Purpose**: Provide execution-proof artifacts for sign-off.
- **Files to create/update**:
  - `tests/evaluation/reports/final_quality_vs_baseline.json` (generated)
  - `tests/evaluation/reports/final_latency_report.json` (generated)
  - `docs/reports/nlu-kg-improvement.md` (final metrics summary)
- **Implementation decisions (fixed)**:
  - Final run must include: NLU tests, KG tests, integration tests, benchmark quality comparison, performance suite.
  - Evidence pack must include command logs and metric deltas.
  - If any quality gate fails, release is blocked regardless of latency gains.
- **Dependencies**: Task D1.
- **QA scenarios (mandatory)**:
  1. Run: `pytest tests/nlu tests/kg tests/integration tests/engine -q`
     - Assert: full pass.
  2. Run: `python -m tests.evaluation.quality_runner --mode compare --against baseline --out tests/evaluation/reports/final_quality_vs_baseline.json`
     - Assert: all quality gates PASS.
  3. Run: `pytest tests/performance -q`
     - Assert: latency/perf reports generated without quality regression.
- **Acceptance criteria**:
  - Final evidence pack complete and auditable for user review.

## Decisions Needed Before Execution
- **[DECISION NEEDED]** Baseline source selection:
  - Option A (default): current main branch behavior as baseline.
  - Option B: manually curated “known-good” checkpoint baseline.
- **[DECISION NEEDED]** Cost tolerance for quality improvements:
  - Option A (default): allow up to +15% token cost during quality-improvement waves if gates improve.
  - Option B: hard cost-neutral constraint.
- **[DECISION NEEDED]** Benchmark language scope:
  - Option A (default): English-only benchmark for current pipeline assumptions.
  - Option B: bilingual benchmark expansion.

## Default Assumptions Applied (until user overrides)
- Baseline = current main branch.
- Cost tolerance = +15% allowed in Waves A/B if quality targets improve.
- Benchmark scope = English-only.



## Final Verification Wave
- Run full test matrix (NLU, KG, integration, evaluation, performance) with archived reports.
- Confirm quality-first gate passed after all latency changes.
- Produce release notes of changed strategies/thresholds/flags.
- Require explicit user "okay" before declaring execution complete.

## Risks and Mitigations
- Risk: overfitting heuristics to benchmark-only examples.
  - Mitigation: include adversarial/unseen cases and maintain mixed scenario set.
- Risk: hidden regressions from threshold tuning.
  - Mitigation: enforce regression dashboard and hard fail thresholds in CI.
- Risk: latency optimization altering KG semantics.
  - Mitigation: use equivalence assertions for pre/post KG state on deterministic fixtures.

## Deliverables
- Single execution plan (this file) with decision-complete tasks.
- Benchmark/evaluation spec and acceptance thresholds.
- Risk log + rollback conditions per high-impact change.
