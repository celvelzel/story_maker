# StoryWeaver Documentation

All project documentation is organized in subdirectories by topic.

## Directory Layout

| Directory | Contents |
|---|---|
| `api/` | Internal module API reference |
| `design/` | Architecture design docs, prompt templates, pipeline diagrams |
| `fixes/` | Bug fix reports and troubleshooting guides |
| `guides/` | Deployment and usage guides |
| `reports/` | Evaluation, optimization, and test reports |
| `project/` | Course project specification PDFs |
| `final_submit/` | Final report and presentation script (academic submission) |

---

## ⭐ Quick Navigation

### 🚀 Getting Started

- **[Zero-to-Hero Deployment](guides/zero-to-hero-deployment.md)** — Full setup guide for all platforms including llama.cpp local inference.
- **[Local Model Startup](guides/local-model-startup.md)** — Quick-start for the llama.cpp server.

### 📖 Core Reference

- [Technical Route](guides/technical-route.md) — NLU/NLG/KG architecture and fallback policies
- [API Reference](api/API_REFERENCE.md) — Module APIs, data structures, configuration

### 🖥️ Platform Deployment

- [Windows Deployment](guides/deployment-windows.md)
- [macOS/Linux Deployment](guides/deployment-macos.md)
- [CPU Inference Guide](guides/CPU_INFERENCE.md)

### 🧪 Testing & Evaluation

- [Automated Test Report](reports/test-results/automated_test_report.md) — NLU/NLG/KG test results
- [KG On/Off Benchmark](reports/test-results/kg-on-off-report.md) — Generation quality with/without KG
- [Tri-Config Comparison](reports/evaluation/tri-config-comparison.md) — API vs. local vs. hybrid

### 🔧 Fixes & Troubleshooting

- [DistilBERT Tokenizer Fix](fixes/distilbert-tokenizer-fix.md)
- [LLM JSON Truncation Fix](fixes/llm-json-truncation-fix.md)
- [FastCoref Fix](fixes/fastcoref-fix.md)
- [FastCoref Activation Analysis](fixes/fastcoref-analysis.md)
- [DistilBERT Compatibility Fix](fixes/distilbert-compatibility-fix.md)
- [DistilBERT Troubleshooting](fixes/distilbert-troubleshooting.md)

### 📐 Architecture & Design

- [Entity Importance Scoring](design/entity-importance.md)
- [Hybrid NLG Architecture](design/hybrid-nlg-architecture.md)
- [KG Summary Modes](design/kg-summary-modes.md)
- [Conflict Detection & Resolution](design/conflict-detection-resolution.md)
- [Sentiment Analysis Design](design/sentiment-analysis.md)
- [NLG Local Model Fine-tuning](design/nlg-local-model-finetuning.md)

### 📈 Knowledge Graph

- [KG Optimization Report](reports/optimization/kg-optimization.md)
- [NLU & KG Improvement Report](reports/optimization/nlu-kg-improvement.md)
- [Runtime Persistence](reports/optimization/runtime-persistence.md)

### 🤖 Local Model Integration

- [Local Inference Integration](reports/local-model/local-inference-integration_2026-03-27.md)
- [Local Model Tuning Report](reports/local-model/local-model-tuning_2026-03-27.md)
- [NLG Local Model Fine-tuning Plan](design/nlg-local-model-finetuning.md)

### 📝 Prompt Engineering

- [Agent Prompt Design](design/prompts/agent_prompt.md)
- [Story Opening Prompt](design/prompts/story_opening.md)
- [Story Continuation Prompt](design/prompts/story_continuation.md)
- [Option Generation Prompt](design/prompts/option_generation.md)

### 📅 Changelog

- [Changelog](reports/changelog/) — Auto-generated update changelogs
