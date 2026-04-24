# Fixes & Troubleshooting

Bug fix reports and troubleshooting guides for issues encountered in StoryWeaver.

## Document Index

### DistilBERT — Intent Classification

- **[distilbert-compatibility-fix.md](distilbert-compatibility-fix.md)** — 5-layer protection scheme for DistilBERT model input compatibility.
- **[distilbert-tokenizer-fix.md](distilbert-tokenizer-fix.md)** — Hardening the DistilBERT tokenizer / classifier interface.
- **[distilbert-troubleshooting.md](distilbert-troubleshooting.md)** — Practical guide for identifying and resolving DistilBERT runtime issues.

### Coreference Resolution — fastcoref

- **[fastcoref-fix.md](fastcoref-fix.md)** — Incompatibility fix between `fastcoref` and `transformers` ≥ 5.2.0.
- **[fastcoref-analysis.md](fastcoref-analysis.md)** — Runtime analysis of fastcoref activation status, rule-based fallback behavior, and re-activation steps.

### LLM Output

- **[llm-json-truncation-fix.md](llm-json-truncation-fix.md)** — Handling truncated JSON outputs from LLMs with structural integrity recovery.

## Recurring Fix Patterns

1. **Input filtering** — Strip unexpected tokenizer output fields based on the model's `forward` signature.
2. **Defensive configuration** — Tighten tokenizer parameters to reduce invalid field generation at source.
3. **Retry with fallback** — All NLU modules implement retry-then-fallback (keyword/rule-based).
4. **Dependency pinning** — Critical library versions are pinned for stable runtime behavior.
5. **Health checks at startup** — Early detection of missing modules or model load failures.
