## Plan: BERT / DistilBERT Intent Replacement and Documentation Delivery

**Goal**: Without breaking the existing pipeline and fallback mechanisms, replace the intent recognition with the CPU-friendly `distilbert-base-uncased`, and fix the NLU model loading lifecycle; then supplement the technical documentation and high-availability deployment documentation, while adding a production one-click script while retaining the existing scripts.

**Steps**
1. Baseline and Impact Confirmation (Phase A)
   - Verify current intent classification inference chain, training chain, configuration entry, and test coverage.
   - Clarify current defects: `GameEngine` instantiated NLU modules but didn't call `load()`, causing models to not take effect (fixed).
   - Constraints: Maintain `rule_fallback` and stable interfaces (`predict() -> {"intent", "confidence"}`).

2. DistilBERT Replacement Design (Phase B)
   - Switch intent model base to `distilbert-base-uncased`.
   - Standardize model directory (e.g., `models/intent_classifier`) and design automatic fallback behavior.
   - Maintain `IntentClassifier` API to avoid affecting `GameEngine` or UI debug display.

3. NLU Loading Lifecycle (Phase C)
   - Explicitly load `coref`, `intent_clf`, and `entity_ext` during `GameEngine` initialization.
   - Add initialization logs and failure degradation logs for observability.
   - Allow configurable model paths in the application entry.

4. Training and Inference Updates (Phase D)
   - Update `training/train_intent.py` with default parameters consistent with DistilBERT.
   - Update dependencies (`transformers`, `torch`) and training/inference instructions in README.
   - Specify recommended batch size, max_length, and resource requirements for CPU scenarios.

5. Testing Enhancement and Regression (Phase E)
   - Add unit tests for "model successful load", "fallback on path failure", and "output structure stability".
   - Update integration tests to cover NLU debug fields in `GameEngine`.

6. Technical Documentation (Phase F)
   - Output technical route documentation: NLU, KG, NLG objectives, I/O, core algorithms, and degradation strategies.
   - Output data flow documentation: turn-by-turn field-level mapping.

7. High-Availability Deployment and Scripts (Phase G)
   - Retain existing scripts: `start_project.bat`, `start_project.sh`.
   - Add production scripts: `start_project_prod.bat`, `start_project_prod.sh`.
   - Production script capabilities: port detection, process identification, safe restart, dependency timeout, environment validation, logging.

8. Acceptance and Verification (Phase H)
   - Verify one-click startup on Windows/macOS.
   - Verify NLU model status in UI/logs.
   - Summarize "Completed / Out of Scope / Recommendations".

**Relevant Files**
- `src/nlu/intent_classifier.py`: Model loading, inference, fallback.
- `src/engine/game_engine.py`: NLU initialization and pipeline calls.
- `app.py`: Engine initialization and UI display.
- `training/train_intent.py`: Training entry for DistilBERT.
- `config.py`: Default model names and runtime configs.
- `scripts/start_project_prod.sh`: New production startup script.

**Verification**
1. Training: Run intent training script and save checkpoint.
2. Inference: Start app, confirm `nlu_debug` output, and verify fallback on model failure.
3. Tests: Run `pytest tests/test_nlu.py` and `pytest tests/test_integration.py`.
4. Deployment: Execute production scripts and verify port detection/restart.
5. Documentation: Verify zero-to-hero deployment steps.

**Decisions**
- Model: `distilbert-base-uncased` (English priority, CPU-friendly).
- Environment: CPU-focused.
- Language: English (updated for consistency).
- Strategy: Add production scripts, retain legacy ones.
- Compatibility: Maintain fallback capabilities.
