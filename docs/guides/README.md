# Guides

Deployment and usage guides for the StoryWeaver project.

## Quick Start Decision Tree

1. **First deployment** → [Zero-to-Hero Deployment Guide](zero-to-hero-deployment.md)
2. **Starting local model only** → [Local Model Startup Guide](local-model-startup.md)
3. **New developer onboarding** → [Technical Route](technical-route.md)
4. **Understanding data flow** → [Data Flow Document](data-flow.md)

## Document Index

### Deployment (by OS)

- **[zero-to-hero-deployment.md](zero-to-hero-deployment.md)** ⭐ Complete from-scratch guide. Covers llama.cpp local model, Windows and macOS.
- **[deployment-windows.md](deployment-windows.md)** — Windows production deployment using `scripts/start/start_project_prod.bat`.
- **[deployment-macos.md](deployment-macos.md)** — macOS/Linux production deployment using `scripts/start/start_project_prod.sh`.
- **[local-model-startup.md](local-model-startup.md)** — Quick-start for the llama.cpp local inference server.

### Architecture & Data Flow

- **[technical-route.md](technical-route.md)** — NLU/NLG/KG architecture, fallback policies, and NLG_MODE routing strategy.
- **[data-flow.md](data-flow.md)** — Field-level data mapping across modules per turn.

### Inference Configuration

- **[CPU_INFERENCE.md](CPU_INFERENCE.md)** — CPU inference optimization guide (legacy; llama.cpp is now preferred for local inference).

## Inference Backend Selection

| Backend | Hardware | Recommended Use |
|---|---|---|
| llama.cpp (local) | CPU / Metal / CUDA | ⭐ Recommended for local. Works on CPU, supports Metal/CUDA. |
| Remote API (Mimo/OpenAI-compat) | Internet connection | Highest quality output, fast prototyping. |
| Hybrid (`NLG_MODE=hybrid`) | Both | Creative tasks on local model, structured tasks via API. |

## Production Launch Commands

**Windows:**
```cmd
scripts\start\start_project_prod.bat
```

**macOS/Linux:**
```bash
chmod +x scripts/start/start_project_prod.sh
./scripts/start/start_project_prod.sh
```

The script auto-creates `.venv`, installs dependencies, handles port conflicts, and writes logs to `logs/storyweaver_prod_<timestamp>.log`.
