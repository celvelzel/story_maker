# Usage Guides

This directory contains usage guides and deployment documentation for the StoryWeaver project.

## Document Overview

### 🚀 Rapid Deployment (Recommended for Beginners)
- **[zero-to-hero-deployment.md](zero-to-hero-deployment.md)** - ⭐ Complete from-scratch deployment guide, including llama.cpp local model inference configuration, compatible with Windows/macOS.
- **[local-model-startup.md](local-model-startup.md)** - Quick start guide for the llama.cpp local server.

### Technical Documentation
- **[technical-route.md](technical-route.md)** - Project technical roadmap, including NLU/NLG/KG architecture design.
- **[data-flow.md](data-flow.md)** - Detailed explanation of data transfer between modules (field-level).

### Deployment Guides (By System)
- **[deployment-windows.md](deployment-windows.md)** - High-availability deployment guide for Windows.
- **[deployment-macos.md](deployment-macos.md)** - High-availability deployment guide for macOS.

### Deprecated / Archived
- **[CPU_INFERENCE.md](CPU_INFERENCE.md)** - ⚠️ DEPRECATED. Legacy vLLM CPU inference guide. Use llama.cpp instead.

## Quick Start

1. **First-time Deployment** → Read the [Zero-to-Hero Deployment Guide](zero-to-hero-deployment.md).
2. **Returning Users** → Use the [Local Model Startup Guide](local-model-startup.md) for quick launches.
3. **New Developers** → Read the [Technical Roadmap](technical-route.md) to understand the overall architecture.
4. **Data Flow** → Check the [Data Flow Document](data-flow.md) for module interaction details.

## Deployment Modes

The project supports two running modes:
- **Development Mode** - Manual launch: `.venv/bin/python -m streamlit run app.py` or `.venv\Scripts\python -m streamlit run app.py`
- **Production Mode** - High-availability deployment using `scripts/start_project_prod.sh` or `scripts/start_project_prod.bat`.

## Inference Backend Selection

| Backend | Hardware Requirements | Recommended Scenario |
|---------|-----------------------|----------------------|
| **llama.cpp (Local)** | CPU / Metal / CUDA | ⭐ Recommended. Works on CPU, supports Metal/CUDA acceleration. |
| **Remote API** | Internet Connection | Fast testing and highest quality output. |