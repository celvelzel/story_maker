# CPU-Only Inference — Deprecated

> **⚠️ DEPRECATED (2026-04-01)**: This document is preserved for historical reference only. The vLLM CPU inference scripts, configuration files, and related infrastructure referenced herein **no longer exist** in this repository.
>
> **Use llama.cpp instead.** See [Local Model Startup Guide](local-model-startup.md) and [Zero-to-Hero Deployment](zero-to-hero-deployment.md).

---

## What Was Removed

The following files and configurations described in the original version of this document have been removed:

| File | Status |
|------|--------|
| `scripts/start_vllm_server_cpu.sh` | ❌ Removed |
| `scripts/start_vllm_server.sh` | ❌ Removed |
| `.env.vllm.cpu` | ❌ Removed |
| `.env.vllm` / `.env.vllm.example` | ❌ Removed |
| `VLLM_INTEGRATION.md` | ❌ Removed |

## Why llama.cpp Replaced vLLM for CPU Inference

| Factor | vLLM (old) | llama.cpp (current) |
|--------|-----------|---------------------|
| CPU support | Experimental, complex setup | Native, out-of-the-box |
| Deployment complexity | High (Python deps, quantization pipeline) | Low (single binary) |
| Memory footprint | Higher | Lower |
| Cross-platform | Linux-focused | Windows/macOS/Linux |
| GGUF support | Limited | Native |
| Metal (Apple Silicon) | No | Yes |
| CUDA | Yes | Yes |

## Migration Path

If you were following this guide, switch to:

1. **[Zero-to-Hero Deployment](zero-to-hero-deployment.md)** — Full setup from scratch
2. **[Local Model Startup](local-model-startup.md)** — Quick llama.cpp server start

### Quick Migration Steps

```bash
# 1. Download llama.cpp binaries
#    https://github.com/ggerganov/llama.cpp/releases
#    Extract to llama.cpp-bin/

# 2. Place GGUF model
mkdir -p models/qwen-gguf/
# Place qwen3-4b-q4_k_m.gguf in models/qwen-gguf/

# 3. Start llama.cpp server
#    Windows: scripts\start_llama_server.bat
#    macOS/Linux: ./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048

# 4. Configure .env
#    OPENAI_BASE_URL=http://127.0.0.1:8081/v1
#    OPENAI_MODEL=qwen3-4b
#    OPENAI_API_KEY=local
```

---

*This document is archived. Last meaningful update: 2026-03-31. Deprecated: 2026-04-01.*
