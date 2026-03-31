# API Documentation

This directory contains the API documentation for the StoryWeaver project.

## Files

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API reference (v1.2.0), including:
  - Frontend interaction models
  - Data structure definitions (TurnResult, StoryOption, GameState, Emotion Result)
  - Backend orchestration (GameEngine with lazy NLU loading, save/load)
  - NLU Module APIs (IntentClassifier, EntityExtractor, CoreferenceResolver, SentimentAnalyzer)
  - NLG Module APIs (StoryGenerator, OptionGenerator, Hybrid NLG Routing)
  - Knowledge Graph APIs (KnowledgeGraph, RelationExtractor, ConflictDetector)
  - Evaluation APIs (8 automatic metrics + 8-dimension LLM Judge)
  - Full configuration reference (all Pydantic Settings parameters)
  - Error handling and usage examples

## Usage

This documentation is the primary reference for frontend-backend integration and internal module communication. It is kept up-to-date with the current codebase state.
