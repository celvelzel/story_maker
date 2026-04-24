# Design Documents

This directory contains technical design documents and architecture specifications for StoryWeaver.

## Document Index

### Core Architecture

- **[entity-importance.md](entity-importance.md)** — Knowledge graph entity importance scoring for pruning. Supports three modes: `composite` (default), `incremental`, and `degree_only`.
- **[nlg-local-model-finetuning.md](nlg-local-model-finetuning.md)** — Local LLM integration and fine-tuning deployment plan for the NLG module.
- **[hybrid-nlg-architecture.md](hybrid-nlg-architecture.md)** — Hybrid NLG routing: task-based routing between local Qwen3 and Mimo Cloud API. Controlled by `NLG_MODE` config.
- **[conflict-detection-resolution.md](conflict-detection-resolution.md)** — Multi-layer conflict detection (rule-based + temporal + LLM) and resolution strategies.
- **[sentiment-analysis.md](sentiment-analysis.md)** — Player input sentiment/emotion analysis design (distilroberta + keyword fallback).
- **[kg-summary-modes.md](kg-summary-modes.md)** — KG summary generation modes: flat vs. layered importance-ranked summary. Controlled by `KG_SUMMARY_MODE`.
- **[implementation_plan.md](implementation_plan.md)** — Original implementation planning document.

### Prompt Templates (`prompts/`)

- **[story_opening.md](prompts/story_opening.md)** — System prompt for generating story openings.
- **[story_continuation.md](prompts/story_continuation.md)** — System prompt for continuing the narrative from player input.
- **[option_generation.md](prompts/option_generation.md)** — Prompt for generating branching player choices.
- **[agent_prompt.md](prompts/agent_prompt.md)** — Agent-level prompt design notes.
- **[gen_doc_prompt.md](prompts/gen_doc_prompt.md)** — Prompt template used to generate project documentation.
- **[gen_ppt_prompt.md](prompts/gen_ppt_prompt.md)** — Prompt template used to generate presentation scripts.

### Pipeline Diagram

- **[storyweaver_pipeline.drawio](storyweaver_pipeline.drawio)** / **[storyweaver_pipeline.svg](storyweaver_pipeline.svg)** — Visual system pipeline diagram.

## Design Principles

1. **Modular** — NLU, NLG, and KG modules are independent and communicate through well-defined interfaces.
2. **Graceful Degradation** — All modules support fallback: keyword NLU, rule-based coreference, flat KG summary.
3. **Configurable** — Key model and strategy choices are hot-switchable via `.env` without code changes.
4. **Narrative Consistency** — World state is maintained through the knowledge graph and conflict detection pipeline.
5. **Hybrid Intelligence** — Local model handles creative tasks; cloud API handles structured extraction; routing is configurable per task type.
