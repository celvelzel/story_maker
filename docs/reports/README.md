# Reports

This directory contains optimization, evaluation, and test reports for the StoryWeaver project.

## Directory Structure

- **optimization/** — System optimization and feature enhancement reports
- **evaluation/** — Model evaluation results (API, local, hybrid, tri-config)
- **local-model/** — Local model inference integration docs
- **test-results/** — Automated test and benchmark outputs
- **changelog/** — Update changelogs (auto-generated)

---

## Optimization (`optimization/`)

- **[kg-optimization.md](optimization/kg-optimization.md)** — Knowledge graph subsystem enhancements: data model, update logic, conflict resolution.
- **[nlu-kg-improvement.md](optimization/nlu-kg-improvement.md)** — 27-item NLU and KG improvement task report.
- **[runtime-persistence.md](optimization/runtime-persistence.md)** — Browser-refresh persistence redesign with `is_active` lifecycle handling.

## Evaluation (`evaluation/`)

- **[api-report.md](evaluation/api-report.md)** — Evaluation results for API-only NLG mode.
- **[local-report.md](evaluation/local-report.md)** — Evaluation results for local model NLG mode.
- **[hybrid-report.md](evaluation/hybrid-report.md)** — Evaluation results for hybrid NLG mode.
- **[tri-config-comparison.md](evaluation/tri-config-comparison.md)** — Side-by-side comparison across all three NLG modes (API / local / hybrid) with zero NLU fallback runs.

## Local Model Integration (`local-model/`)

- **[local-inference-integration_2026-03-27.md](local-model/local-inference-integration_2026-03-27.md)** — Initial llama.cpp CPU inference integration.
- **[local-model-tuning_2026-03-27.md](local-model/local-model-tuning_2026-03-27.md)** — Local model tuning: logging and timeout configuration.

## Test Results (`test-results/`)

- **[automated_test_report.md](test-results/automated_test_report.md)** — Automated test results for NLU/NLG/KG modules.
- **[kg-on-off-report.md](test-results/kg-on-off-report.md)** — KG on/off benchmark comparing generation quality with and without the knowledge graph.

## Changelog (`changelog/`)

- **[changelog_2026-03-24_initial.md](changelog/changelog_2026-03-24_initial.md)** — Initial core loop and architecture setup.
- **[changelog_2026-03-24_error-logging.md](changelog/changelog_2026-03-24_error-logging.md)** — Error handling and logging system.
- **[changelog_2026-03-25_eval-metrics-expansion.md](changelog/changelog_2026-03-25_eval-metrics-expansion.md)** — Distinct-n / Self-BLEU evaluation metrics expansion.
