# NLG Module Local LLM Fine-tuning and Deployment Plan

> Project: StoryWeaver (text adventure game)
> 
> Goal: Migrate the NLG module from a single cloud API call to a fine-tuned LLM capable of running on a local machine (e.g., AMD R7 without dedicated GPU). Support seamless switching between the local model and cloud API to compare performance.

---

## 1. Current Architecture Analysis

StoryWeaver's current NLG pipeline:

1.  **UI Layer**: `app.py` (Streamlit) handles rendering.
2.  **Engine Layer**: `src/nlg/story_generator.py` and `src/nlg/option_generator.py` construct prompts.
3.  **Prompt Templates**: `docs/design/prompts/` define strict I/O formats (including `kg_summary`, `history`, `intent`, etc.).
4.  **API Client**: `src/utils/api_client.py` uses a **singleton pattern** to wrap an OpenAI-compatible client.
5.  **Configuration**: `config.py` (Pydantic Settings) manages `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`, loading from `.env`.
6.  **Current Model**: Uses `gpt-4o-mini` or compatible models via OpenAI-compatible endpoints.

**Key Finding**: The project supports OpenAI-compatible interfaces. We extended `config.py` and `src/utils/api_client.py` for dynamic switching to enable hot-swapping between local and cloud models.

---

## 2. Hardware Environment and Model Selection

### 2.1 Local Hardware
*   **CPU**: AMD R7 (or similar modern mobile/desktop processor)
*   **RAM**: 16-32 GB
*   **Inference**: Pure CPU inference or acceleration via Vulkan/ROCm using integrated graphics (Ollama/llama.cpp).

### 2.2 Model Selection: Llama-3.2-3B-Instruct / Qwen2.5-3B
*   **Parameter Count**: 3B.
*   **Advantages**: Excellent instruction following, small footprint, suitable for structured output in interactive fiction.
*   **Quantization**:
    *   **Q8_0**: 8-bit (approx. 3.5GB RAM), near-lossless precision.
    *   **Q4_K_M**: 4-bit (approx. 2.0GB RAM), faster with minor precision loss.
*   **Expected Performance**: 20-30 tokens/s on modern CPUs, providing a fluid experience.

---

## 3. Fine-tuning Platform: PolyU Student HPC

Training requires GPUs and is conducted on the PolyU Student HPC.

### 3.1 HPC Environment Setup
1.  Login via SSH.
2.  Request an interactive GPU node: `srun -p gpu --gres=gpu:1 --pty bash`
3.  Load modules:
    ```bash
    module load anaconda3
    module load cuda/12.1
    ```
4.  Create environment:
    ```bash
    conda create -n swift_env python=3.10 -y
    conda activate swift_env
    pip install ms-swift[llm]
    ```

---

## 4. Dataset Strategy: Prompt Synthesis

### 4.1 Approach: Bulk Synthesis via LLM
Use the existing cloud API (`gpt-4o-mini`) to generate `(Prompt → Response)` pairs.

*   **Advantages**:
    * 100% alignment with project `prompt_templates`.
    * Fast and diverse data generation by mixing templates for `kg_summary`, `history`, `intent`, and `emotion`.
*   **Implementation**: `training/train_generator.py` and `training/data_augmenter.py`.

### 4.2 Data Volume
*   **Story Generation**: 300 samples (Opening + Continuation).
*   **Option Generation**: 300 samples (JSON structured).
*   **Total**: 600+ samples, stored in `training/nlg_dataset/combined_data.jsonl`.

### 4.3 Format (ChatML)
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert interactive-fiction narrator..."},
    {"role": "user", "content": "kg_summary: ...\nhistory: ...\nintent: explore\nplayer_input: go north"},
    {"role": "assistant", "content": "You step towards the north, the cold wind bites your skin..."}
  ]
}
```

---

## 5. Fine-tuning: ms-swift + LoRA

### 5.1 Training Scripts
Training is automated via shell scripts in the `training/` directory:
- `training/train_llama.sh`: Fine-tunes Llama-3.2-3B.
- `training/train_qwen.sh`: Fine-tunes Qwen-2.5-3B.

### 5.2 Configuration
*   **Method**: LoRA (Low-Rank Adaptation).
*   **Command Example**:
    ```bash
    swift sft \
        --model_type llama3_2-3b-instruct \
        --dataset training/nlg_dataset/combined_data.jsonl \
        --sft_type lora \
        --output_dir output/nlg_model \
        --learning_rate 2e-4 \
        --num_train_epochs 3 \
        --batch_size 4
    ```

---

## 6. Export and Quantization

After training, weights are merged and exported to GGUF format for local CPU inference.

```bash
# Merge LoRA
swift export --model_type llama3_2-3b-instruct --adapters output/nlg_model/v0-... --merge_lora true

# Export GGUF (Q8_0)
swift export --model_type llama3_2-3b-instruct --model_id_or_path <merged_path> --to_gguf true --quant_bits 8 --quant_method q8_0
```

---

## 7. Local Deployment (llama.cpp / Ollama)

The project supports both `llama.cpp` and `Ollama` for local inference.

1.  **llama.cpp**: Start the server using `scripts/start_llama_server.sh`.
2.  **Ollama**: Create a `Modelfile` and run `ollama create storyweaver-model -f Modelfile`.

Local models are served at an OpenAI-compatible endpoint (e.g., `http://localhost:8000/v1` or `http://localhost:11434/v1`).

---

## 8. UI Integration

A toggle in the Streamlit sidebar allows switching between:
*   **Cloud API** (OpenAI)
*   **Local Model** (llama.cpp/Ollama)

Switching triggers a configuration update in `src/utils/api_client.py` to route requests to the selected backend.

---

## 9. Roadmap Status

1. [x] Dataset generation script (`training/train_generator.py`)
2. [x] Training scripts for Llama and Qwen (`training/train_*.sh`)
3. [x] Model Export (GGUF integration)
4. [x] Local Deployment scripts (`scripts/start_llama_server.sh`)
5. [x] UI Dynamic Switching Implementation


