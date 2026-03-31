# Design Documentation

This directory contains technical design documents and specifications for the StoryWeaver project.

## Document Index

### Core Architecture
- **[entity-importance.md](entity-importance.md)** - Entity importance scoring strategy for Knowledge Graph pruning. Supports `composite` and `degree_only` modes.
- **[nlg-local-model-finetuning.md](nlg-local-model-finetuning.md)** - Fine-tuning and deployment plan for local LLM integration in the NLG module.

### Prompt Templates (`prompts/`)
- **[story_opening.md](prompts/story_opening.md)** - System prompts for generating new story beginnings.
- **[story_continuation.md](prompts/story_continuation.md)** - System prompts for continuing the narrative based on player input.
- **[option_generation.md](prompts/option_generation.md)** - Prompts for generating branching player choices.

## Design Principles

1.  **Modular Architecture** - NLU, NLG, and KG modules are independent and communicate via well-defined interfaces.
2.  **Graceful Degradation** - All modules support fallback mechanisms (e.g., keyword-based NLU if models fail) to ensure continuous service.
3.  **Extensibility** - Support for hot-swapping models and dynamic strategy adjustments via configuration.
4.  **World Consistency** - State management via Knowledge Graph and Conflict Detection to maintain narrative integrity.
