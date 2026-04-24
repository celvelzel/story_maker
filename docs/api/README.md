# API Reference

Internal module API documentation for StoryWeaver.

## Files

- **[API_REFERENCE.md](API_REFERENCE.md)** — Full API reference (v1.2.0), covering:
  - Frontend interaction model (TurnResult, StoryOption, GameState, EmotionResult)
  - Game engine orchestrator (`GameEngine` with lazy NLU loading, save/load)
  - NLU module APIs: `IntentClassifier`, `EntityExtractor`, `CoreferenceResolver`, `SentimentAnalyzer`
  - NLG module APIs: `StoryGenerator`, `OptionGenerator`, hybrid NLG routing (`NLG_MODE`)
  - Knowledge graph APIs: `KnowledgeGraph`, `RelationExtractor`, `ConflictDetector`
  - Evaluation APIs: 8 automatic metrics + 8-dimension LLM judge scoring
  - Runtime session persistence: `save_runtime_session`, `load_runtime_session`, `is_active` lifecycle flag
  - Full configuration reference (all Pydantic Settings parameters)
  - Error handling patterns and usage examples

## Usage Note

This document is the primary reference for internal module integration. It is kept in sync with the codebase.
