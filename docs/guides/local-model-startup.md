# Local Model Inference Startup Guide

This document describes how to start the llama.cpp server locally and run the StoryWeaver application.

> **Last Updated**: 2026-04-01

## Prerequisites

- [llama.cpp](https://github.com/ggerganov/llama.cpp) binaries (placed in `llama.cpp-bin/`)
- GGUF model file in `models/qwen-gguf/`
- Windows/macOS/Linux environment with Python 3.10+

## 1. Quick Start

### Windows

```powershell
# Step 1: Start the local llama.cpp server (Open a new terminal window)
.\scripts\start_llama_server.bat

# Step 2: Start the StoryWeaver application (Open another new terminal window)
.\scripts\start_project_prod.bat
```

### macOS / Linux

```bash
# Step 1: Start the local llama.cpp server (Open a new terminal window)
# Option A: Use the batch script wrapper (if available)
# Option B: Run llama-server directly:
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml

# Step 2: Start the StoryWeaver application (Open another new terminal window)
./scripts/start_project_prod.sh
```

### Access URLs

- **StoryWeaver Application**: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **llama.cpp API Endpoint**: [http://127.0.0.1:8081/v1/chat/completions](http://127.0.0.1:8081/v1/chat/completions)

## 2. Startup Script Details

### start_llama_server.bat (Windows)

Starts the local llama.cpp server, providing an OpenAI-compatible API.

**Configuration Parameters** (Adjustable within the script):

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `LLAMA_BIN` | `llama.cpp-bin\llama-server.exe` | Path to the llama-server executable |
| `MODEL_PATH` | `models\qwen-gguf\qwen3-4b-q4_k_m.gguf` | Path to the GGUF quantized model |
| `PORT` | `8081` | API service port |
| `HOST` | `127.0.0.1` | Listen address |
| `CONTEXT_SIZE` | `2048` | Context length (tokens) |
| `BATCH_SIZE` | `512` | Batch size |
| `THREADS` | `4` | Number of CPU threads to use |

**Expected Console Output**:

```text
=========================================
  llama.cpp Local API Server
=========================================

[INFO] Model:    models\qwen-gguf\qwen3-4b-q4_k_m.gguf
[INFO] Server:   http://127.0.0.1:8081
[INFO] Context:  2048 tokens
[INFO] Threads:  4

OpenAI-compatible endpoints:
  Chat:    http://127.0.0.1:8081/v1/chat/completions
  Models:  http://127.0.0.1:8081/v1/models
```

### macOS/Linux: Direct llama-server Command

There is no `start_llama_server.sh` script. Run `llama-server` directly:

**CPU mode:**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml
```

**Apple Silicon Metal acceleration (recommended):**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 8 --ngl 99 --chat-template chatml
```

**NVIDIA CUDA acceleration:**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 4096 -b 512 -t 8 --ngl 99 --chat-template chatml
```

### start_project_prod (bat/sh)

Starts the StoryWeaver Streamlit application in production mode.

**Features**:
- Automatically detects and handles processes occupying the target port.
- Creates a virtual environment and installs dependencies if missing.
- Redirects logs to the `logs/` directory.

## 3. Required Files

Ensure the following files are present:

1. **llama-server**: `llama.cpp-bin/llama-server` (or `.exe` on Windows)
2. **Quantized Model**: `models/qwen-gguf/qwen3-4b-q4_k_m.gguf`
3. **Configuration**: `.env` (configured for local backend)

### Verify Model Files

```bash
ls -lh models/qwen-gguf/
```

Expected output:
- `qwen3-4b-q4_k_m.gguf` - Q4 quantized model (~2.4GB, recommended)

## 4. Environment Configuration (.env)

The project supports three NLG modes via `config.py` (`NLG_MODE`): `api`, `local`, `hybrid`.

**For local llama.cpp backend**, configure `.env` as follows:

```ini
# llama.cpp Local Inference
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_MODEL=qwen3-4b
OPENAI_API_KEY=local

# Timeout (adjust based on hardware)
# CPU inference: longer timeouts
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0

# GPU (CUDA/Metal): shorter timeouts
# OPENAI_TIMEOUT_CONNECT=10.0
# OPENAI_TIMEOUT_READ=60.0

OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

To switch to remote API, update `OPENAI_BASE_URL` and `OPENAI_MODEL` accordingly. See `config/.env.example` for the remote API template.

## 5. Live Logs

When the application sends an LLM request, the llama.cpp server terminal will display request processing information.

## 6. Troubleshooting

### Port Already in Use

**Windows**:
```powershell
netstat -ano | findstr :8081
taskkill /PID <PID> /F
```

**macOS/Linux**:
```bash
lsof -i :8081
kill -9 <PID>
```

### Model Not Found

Ensure the model file is in `models/qwen-gguf/`. See [Zero-to-Hero Deployment](zero-to-hero-deployment.md) for model download instructions.

### Connection Failed

1. Check if the llama.cpp server is actually running.
2. Verify port `8081` is accessible: `curl http://127.0.0.1:8081/v1/models`
3. Confirm `OPENAI_BASE_URL` in `.env` matches the server address.

## 7. Performance Tips

- **Quantization**: Use `Q4_K_M` for a good balance of speed and quality (uses ~2.4GB RAM).
- **Threads**: Match the `--threads` parameter to your physical CPU core count.
- **Context Size**: Increase `--ctx-size` if you plan to have very long story sessions.
- **Metal/CUDA**: Use `--ngl 99` to offload all layers to GPU for significant speedup.

## 8. Related Documents

- [Zero-to-Hero Deployment Guide](zero-to-hero-deployment.md)
- [CPU Inference (Deprecated)](CPU_INFERENCE.md)

---
*Last Updated: 2026-04-01*
