# NLG Models Directory

This directory stores fine-tuned Large Language Models (LLMs) used for local inference, providing an alternative to third-party API calls.

## Purpose

- **Local Inference:** Replace OpenAI-compatible API calls with locally hosted models.
- **Data Privacy:** Keep sensitive story data within the local infrastructure.
- **Offline Development:** Enable story and option generation without an internet connection.

## Configuration

The project interacts with LLMs via the `src/nlg/` module. To switch to local models, update `config.py` or your `.env` file:

```bash
# .env Configuration Example
NLG_MODE=local
OPENAI_BASE_URL=http://localhost:8000/v1  # Address of your local inference server (e.g., vLLM or llama.cpp)
OPENAI_MODEL=qwen3-4b-instruct            # The model name as registered on the server
```

## Supported Formats

- **vLLM:** Recommended for high-performance serving. Use `vllm serve models/nlg/your-model/`.
- **Hugging Face:** Standard transformers format.
- **GGUF:** Optimized for CPU/GPU inference via `llama.cpp`.

## Deployment Example

### Local Serving with vLLM

```bash
vllm serve models/nlg/your-model/ \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

### API Compatibility

The project's NLG module uses an OpenAI-compatible interface. Any local server that follows this standard (like vLLM, Ollama, or llama.cpp's server) can be used by updating the `OPENAI_BASE_URL` and `OPENAI_MODEL` settings.

