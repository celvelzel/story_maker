# Fixes and Troubleshooting

This directory documents the fixes and troubleshooting solutions for various issues encountered in the StoryWeaver project.

## Document Index

### DistilBERT (Intent Classification)
- **[DistilBERT Compatibility Fix](distilbert-compatibility-fix.md)** - Comprehensive fix for DistilBERT model input compatibility, implementing a 5-layer protection scheme.
- **[DistilBERT Tokenizer Fix](distilbert-tokenizer-fix.md)** - Technical report on hardening the interface between the DistilBERT tokenizer and the classification model.
- **[DistilBERT Troubleshooting](distilbert-troubleshooting.md)** - Practical guide for identifying and resolving DistilBERT-related runtime issues.

### Coreference Resolution
- **[FastCoref Fix](fastcoref-fix.md)** - Resolution for incompatibilities between `fastcoref` and newer `transformers` versions (specifically 5.2.0+).

### LLM & Generation
- **[LLM JSON Truncation Fix](llm-json-truncation-fix.md)** - Strategy for handling truncated JSON outputs from LLMs and ensuring structural integrity.

## Common Fix Patterns

We employ several recurring patterns to ensure system stability:

1.  **Automatic Input Filtering**: Automatically stripping unexpected fields from tokenizer outputs based on the model's `forward` signature.
2.  **Defensive Configuration**: Reducing invalid field generation at the source by tightening tokenizer parameters.
3.  **Safety Wrappers**: Implementing retry-with-fallback mechanisms and version-aware warning systems.
4.  **Dependency Pinning**: Locking critical library versions to ensure a stable testing and runtime environment.
5.  **Proactive Health Checks**: Early detection of environmental or model loading issues during startup.