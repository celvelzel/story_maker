# NLG 模块本地 LLM 微调与部署计划

> **最后更新：** 2026-04-01  
> 项目：StoryWeaver（文字冒险游戏）
> 
> 目标：将 NLG 模块从单一云端 API 调用迁移到可在本地机器（如 AMD R7 无独立 GPU）运行的微调 LLM。支持在本地模型和云端 API 之间无缝切换以对比性能。

---

## 1. 当前架构分析

StoryWeaver 当前的 NLG 流水线：

1.  **UI 层**：`app.py`（Streamlit）负责渲染
2.  **引擎层**：`src/nlg/story_generator.py` 和 `src/nlg/option_generator.py` 构建提示词
3.  **提示词模板**：`src/nlg/prompt_templates.py` 定义所有模板（`SYSTEM_PROMPT`、`OPENING_PROMPT`、`STORY_CONTINUE_PROMPT`、`OPTION_GENERATION_PROMPT`）
4.  **API 客户端**：`src/utils/api_client.py` 使用**多类型单例模式**，通过 `LLMClient` 和 `HybridClientManager` 实现基于任务的路由
5.  **配置**：`config.py`（Pydantic Settings）管理 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 和 `NLG_MODE`，从 `.env` 加载
6.  **当前模型**：默认 `mimo-v2-flash`，通过 OpenAI 兼容端点访问。混合模式将创意任务（故事）路由到本地 Qwen3，结构化任务（选项、关系）路由到 Mimo API

**关键发现**：架构支持三种 NLG 模式（`api`、`local`、`hybrid`），使用按类型的 LLM 客户端单例。`HybridClientManager` 路由任务：故事 → 本地，选项/关系/JSON → API

---

## 2. 硬件环境与模型选择

### 2.1 本地硬件
*   **CPU**：AMD R7（或类似的现代移动/桌面处理器）
*   **内存**：16-32 GB
*   **推理**：纯 CPU 推理或通过集成显卡使用 Vulkan/ROCm 加速（Ollama/llama.cpp）

### 2.2 模型选择：Llama-3.2-3B-Instruct / Qwen2.5-3B
*   **参数量**：3B
*   **优势**：出色的指令遵循能力，占用空间小，适合交互式小说的结构化输出
*   **量化**：
    *   **Q8_0**：8 位（约 3.5GB 内存），近乎无损精度
    *   **Q4_K_M**：4 位（约 2.0GB 内存），速度更快，精度略有损失
*   **预期性能**：现代 CPU 上 20-30 tokens/s，提供流畅体验

---

## 3. 微调平台：PolyU 学生 HPC

训练需要 GPU，在 PolyU 学生 HPC 上进行。

### 3.1 HPC 环境设置
1.  通过 SSH 登录
2.  请求交互式 GPU 节点：`srun -p gpu --gres=gpu:1 --pty bash`
3.  加载模块：
    ```bash
    # 加载 Anaconda3 环境
    module load anaconda3
    # 加载 CUDA 12.1 工具链
    module load cuda/12.1
    ```
4.  创建环境：
    ```bash
    # 创建 Python 3.10 环境
    conda create -n swift_env python=3.10 -y
    conda activate swift_env
    # 安装 ms-swift 微调框架
    pip install ms-swift[llm]
    ```

---

## 4. 数据集策略：提示词合成

### 4.1 方法：通过 LLM 批量合成
使用现有云端 API（`gpt-4o-mini`）生成 `(Prompt → Response)` 配对数据。

*   **优势**：
    * 与项目 `prompt_templates` 100% 对齐
    * 通过混合 `kg_summary`、`history`、`intent` 和 `emotion` 模板，快速生成多样化数据
*   **实现**：`training/train_generator.py` 和 `training/data_augmenter.py`

### 4.2 数据量
*   **故事生成**：300 条样本（开场 + 续写）
*   **选项生成**：300 条样本（JSON 结构化）
*   **总计**：600+ 条样本，存储在 `training/nlg_dataset/combined_data.jsonl`

### 4.3 格式（ChatML）
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert interactive-fiction narrator..."},  # 系统提示：你是交互式小说叙述专家
    {"role": "user", "content": "kg_summary: ...\nhistory: ...\nintent: explore\nplayer_input: go north"},  # 用户输入：包含知识图谱摘要、历史、意图和玩家输入
    {"role": "assistant", "content": "You step towards the north, the cold wind bites your skin..."}  # 助手回复：生成的叙事文本
  ]
}
```

---

## 5. 微调：ms-swift + LoRA

### 5.1 训练脚本
训练通过 `training/` 目录中的 shell 脚本自动化：
- `training/train_llama.sh`：微调 Llama-3.2-3B
- `training/train_qwen.sh`：微调 Qwen-2.5-3B

### 5.2 配置
*   **方法**：LoRA（低秩适配）
*   **命令示例**：
    ```bash
    swift sft \
        --model_type llama3_2-3b-instruct \  # 指定模型类型
        --dataset training/nlg_dataset/combined_data.jsonl \  # 指定训练数据集
        --sft_type lora \  # 使用 LoRA 微调方式
        --output_dir output/nlg_model \  # 输出目录
        --learning_rate 2e-4 \  # 学习率
        --num_train_epochs 3 \  # 训练轮数
        --batch_size 4  # 批次大小
    ```

---

## 6. 导出与量化

训练完成后，权重合并并导出为 GGUF 格式，用于本地 CPU 推理。

```bash
# 合并 LoRA 权重
swift export --model_type llama3_2-3b-instruct --adapters output/nlg_model/v0-... --merge_lora true

# 导出为 GGUF 格式（Q8_0 量化）
swift export --model_type llama3_2-3b-instruct --model_id_or_path <merged_path> --to_gguf true --quant_bits 8 --quant_method q8_0
```

---

## 7. 本地部署（llama.cpp / Ollama）

项目支持 `llama.cpp` 和 `Ollama` 两种本地推理方式。

1.  **llama.cpp**：使用 `scripts/start_llama_server.sh` 启动服务器
2.  **Ollama**：创建 `Modelfile` 并运行 `ollama create storyweaver-model -f Modelfile`

本地模型通过 OpenAI 兼容端点提供服务（如 `http://localhost:8000/v1` 或 `http://localhost:11434/v1`）

---

## 8. UI 集成

Streamlit 侧边栏中的切换开关允许在以下模式之间切换：
*   **云端 API**（OpenAI）
*   **本地模型**（llama.cpp/Ollama）

切换会触发 `src/utils/api_client.py` 中的配置更新，将请求路由到选定的后端

---

## 9. 路线图状态

1. [x] 数据集生成脚本（`training/train_generator.py`）
2. [x] Llama 和 Qwen 训练脚本（`training/train_*.sh`）
3. [x] 模型导出（GGUF 集成）
4. [x] 本地部署脚本（`scripts/start_llama_server.sh`）
5. [x] UI 动态切换实现
