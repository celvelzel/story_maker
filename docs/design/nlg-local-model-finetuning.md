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

**Key Finding**: The project already supports OpenAI-compatible interfaces. We only need to extend `config.py` and `src/utils/api_client.py` for dynamic switching and cache resetting to enable hot-swapping between local and cloud models.

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

Training requires GPUs and will be conducted on the PolyU Student HPC.

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
*   **Implementation**: `training/generate_dataset.py`.

### 4.2 Data Volume
*   **Story Generation**: 300 samples (Opening + Continuation).
*   **Option Generation**: 300 samples (JSON structured).
*   **Total**: 600 samples (adjustable).

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

### 5.1 Configuration
*   **Method**: LoRA (Low-Rank Adaptation).
*   **Command**:
    ```bash
    swift sft \
        --model_type llama3_2-3b-instruct \
        --dataset /path/to/nlg_dataset.jsonl \
        --sft_type lora \
        --output_dir output/nlg_model \
        --learning_rate 2e-4 \
        --num_train_epochs 3 \
        --batch_size 4
    ```

---

## 6. Export and Quantization

After training, merge weights and export to GGUF format for local CPU inference.

```bash
# Merge LoRA
swift export --model_type llama3_2-3b-instruct --adapters output/nlg_model/v0-... --merge_lora true

# Export GGUF (Q8_0)
swift export --model_type llama3_2-3b-instruct --model_id_or_path <merged_path> --to_gguf true --quant_bits 8 --quant_method q8_0
```

---

## 7. Local Deployment (Ollama)

1.  Create a `Modelfile`:
    ```text
    FROM ./model-q8_0.gguf
    ```
2.  Create model: `ollama create storyweaver-model -f Modelfile`
3.  Ollama provides an OpenAI-compatible API at `http://localhost:11434/v1`.

---

## 8. UI Integration

Add a toggle in the Streamlit sidebar to switch between:
*   **Cloud API**
*   **Local Model**

Updating the selection will trigger `api_client.reload_client()` to refresh the connection.

---

## 9. Roadmap

1. [x] Dataset generation script (`training/generate_dataset.py`)
2. [ ] HPC Training (SLURM)
3. [ ] Model Export (GGUF)
4. [ ] Local Testing (Ollama)
5. [ ] UI Dynamic Switching Implementation

