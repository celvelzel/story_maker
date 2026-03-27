# vLLM Integration Guide

This guide explains how to deploy and integrate a local vLLM server to serve the merged Qwen model as an OpenAI-compatible API for the StoryWeaver project.

## Purpose

By deploying a local vLLM server, you can swap the project's story generation and option generation backends from the cloud-based OpenAI API to a locally hosted model. This ensures data privacy, reduces API costs, and allows for offline development using your fine-tuned models.

## Step 1: Configuration

First, set up the environment variables for the vLLM server.

1. Copy the example environment file:
   ```bash
   cp .env.vllm.example .env.vllm
   ```
2. Open `.env.vllm` and modify the `MODEL_PATH` to point to your merged model directory:
   ```env
   MODEL_PATH=/path/to/your/merged_model
   ```
   *Note: Ensure the path is absolute or relative to the project root.*

## Step 2: Start the vLLM Server

Run the provided startup script to launch the server. It is recommended to run this in a `tmux` session or using `nohup` if you are on an HPC cluster to keep the server running in the background.

```bash
# Make the script executable if needed
chmod +x scripts/start_vllm_server.sh

# Start the server
./scripts/start_vllm_server.sh
```

The server will start on `http://127.0.0.1:8000` by default (as configured in `.env.vllm`).

## Step 3: Verify the Deployment

Once the server is running, verify that it correctly handles OpenAI-compatible requests using the test script:

```bash
python scripts/test_openai_api.py
```

If successful, you should see standard and streaming completions printed in your terminal.

## Step 4: Update Project Configuration (CRITICAL)

To make the main StoryWeaver application use your local vLLM server instead of the official OpenAI API, you must update the project's main `.env` file.

1. Open the `.env` file in the project root.
2. Update the following variables to match your local server configuration:

   ```env
   OPENAI_API_KEY=sk-local-test
   OPENAI_BASE_URL=http://127.0.0.1:8000/v1
   ```

   *Note: The `/v1` suffix is required by the OpenAI Python client to correctly route requests to the vLLM endpoint.*

## How to Revert

If you need to switch back to the original OpenAI API, simply update your `.env` file again:

```env
OPENAI_API_KEY=your-original-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

No code changes are required, as the project uses a unified API client that respects these environment variables.
