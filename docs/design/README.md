# Design Documentation

This directory contains technical design documents and specifications for the StoryWeaver project.

## Document Index

### Core Architecture
- **[entity-importance.md](entity-importance.md)** - Entity importance scoring strategy for Knowledge Graph pruning. Supports `composite`, `incremental`, and `degree_only` modes.
- **[nlg-local-model-finetuning.md](nlg-local-model-finetuning.md)** - Fine-tuning and deployment plan for local LLM integration in the NLG module.
- **[hybrid-nlg-architecture.md](hybrid-nlg-architecture.md)** - Hybrid NLG routing architecture: local Qwen3 + Mimo API task-based routing.
- **[conflict-detection-resolution.md](conflict-detection-resolution.md)** - Multi-layer conflict detection (rule + temporal + LLM) and resolution strategies.
- **[sentiment-analysis.md](sentiment-analysis.md)** - Emotion/sentiment analysis design for player input (distilroberta + keyword fallback).
- **[kg-summary-modes.md](kg-summary-modes.md)** - KG summary generation modes: flat vs layered importance-ranked summaries.

### Prompt Templates (`prompts/`)
- **[story_opening.md](prompts/story_opening.md)** - System prompts for generating new story beginnings.
- **[story_continuation.md](prompts/story_continuation.md)** - System prompts for continuing the narrative based on player input.
- **[option_generation.md](prompts/option_generation.md)** - Prompts for generating branching player choices.

## Design Principles

1.  **Modular Architecture** - NLU, NLG, and KG modules are independent and communicate via well-defined interfaces.
2.  **Graceful Degradation** - All modules support fallback mechanisms (e.g., keyword-based NLU if models fail) to ensure continuous service.
3.  **Extensibility** - Support for hot-swapping models and dynamic strategy adjustments via configuration.
4.  **World Consistency** - State management via Knowledge Graph and Conflict Detection to maintain narrative integrity.
5.  **Hybrid Intelligence** - Local models for creative tasks, cloud API for structured tasks, optimizing cost and quality.
