# NLG 模块本地大模型微调与部署方案

> 项目：StoryWeaver (text adventure game)
> 
> 目标：将 NLG 模块从单一的云端 API 调用，迁移至可在本地 AMD R7 (无 GPU) 机器上运行的微调大模型，并支持用户在本地模型与云端 API 之间自由切换比较性能。

---

## 1. 现有架构分析

经过对项目代码的审查，StoryWeaver 当前的 NLG 调用链路如下：

1.  **UI 层**：`app.py` (Streamlit) 负责渲染。
2.  **引擎层**：`src/nlg/story_generator.py` 和 `src/nlg/option_generator.py` 负责构建 Prompt。
3.  **Prompt 模板**：`src/nlg/prompt_templates.py` 定义了严格的输入输出格式（包含 `kg_summary`, `history`, `intent` 等变量）。
4.  **API 封装层**：`src/utils/api_client.py` 通过**单例模式**封装了 OpenAI 兼容的客户端。
5.  **配置层**：`config.py` (Pydantic Settings) 统一管理 `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` 等参数，支持从 `.env` 文件加载。
6.  **当前模型**：使用的是 `mimo-v2-flash` 模型（通过配置的兼容 OpenAI 的 API 端点访问）。

**关键发现**：项目底层已经完美支持 OpenAI 兼容接口。这意味着我们**无需重写任何核心逻辑代码**，只需在 `config.py` 和 `src/utils/api_client.py` 基础上做微小扩展（如运行时动态切换、缓存重置），即可实现本地模型与云端 API 的热切换。

---

## 2. 硬件环境与模型选型

### 2.1 本地硬件条件
*   **CPU**: AMD R7 (H255 系列移动端处理器，无独立显卡，自带核显)
*   **内存 (RAM)**: 32 GB
*   **推理方式**：纯 CPU 推理（或通过 Ollama/llama.cpp 启用 Vulkan/ROCm 利用核显加速）

### 2.2 选型：Llama-3.2-3B-Instruct
*   **参数量**：3B（30 亿参数）。
*   **优势**：在 3B 级别中表现顶尖，英文指令遵循能力优秀，适合交互式文本游戏的结构化输出。
*   **量化方案**：
    *   由于有 32GB 内存，我们将优先采用 **Q8_0 量化**（8-bit，约占用 3.5GB 内存），接近无损精度。
    *   如需更极致的速度，可降级为 **Q4_K_M 量化**（约占用 2.0GB 内存），速度更快但略有精度损失。
*   **预期性能**：在 AMD R7 纯 CPU 环境下，生成速度预计可达 **20-30 tokens/s**，体验流畅无明显延迟。

---

## 3. 微调平台：PolyU Student HPC

将使用香港理工大学提供的 Student HPC 平台进行模型微调（训练过程需要 GPU，不适合在本地完成）。

### 3.1 平台规范
*   调度系统：**SLURM**
*   作业提交方式：`sbatch`（批量任务脚本）
*   环境管理：`module load` 加载 Anaconda 和 CUDA，使用 Conda 虚拟环境

### 3.2 HPC 环境准备流程
1.  通过 SSH 登录 HPC。
2.  申请交互式 GPU 节点：`srun -p gpu --gres=gpu:1 --pty bash`
3.  加载模块：
    ```bash
    module load anaconda3
    module load cuda/12.1   # 具体版本参考 module avail
    ```
4.  创建并安装环境：
    ```bash
    conda create -n swift_env python=3.10 -y
    conda activate swift_env
    pip install ms-swift[llm]
    ```

---

## 4. 数据集方案：批量 Prompt 合成

### 4.1 为何不使用网上现成语料库？
StoryWeaver 有高度结构化的输入输出规范（`kg_summary`, `history`, `intent`, JSON 格式的选项等）。网上没有任何现成数据集能与此格式对齐，强行使用需要巨大的清洗成本且难以生成合理的知识图谱上下文。

### 4.2 方案：批量构造 Prompt 直接调用 LLM
利用项目中已配置好的 `mimo-v2-flash` 云端 API，批量发送构造好的多样化 prompt，直接收集 `(Prompt → Response)` 对作为训练数据。

*   **优点**：
    * 数据格式与项目 prompt_templates 100% 对齐，无需清洗
    * 无需启动 GameEngine，速度极快，数百条数据几分钟即可完成
    * 通过组合预写的 kg_summary、history、genre、intent、emotion 等模板，可批量生成海量多样化上下文
*   **成本**：`mimo-v2-flash` API 成本极低，生成 600 条数据几乎免费
*   **实现方式**：`scripts/generate_dataset.py`（Prompt-based 版本）

### 4.3 数据生成量
默认配置生成：
*   **Story Generation**: 300 条（~60 条 opening + ~240 条 continuation）
*   **Option Generation**: 300 条（JSON 格式的结构化选项）
*   **总计**: 600 条，可通过 `--story` 和 `--option` 参数调节

### 4.4 数据格式
采用 `ms-swift` 标准的 OpenAI ChatML 格式（`.jsonl`）：
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert interactive-fiction narrator..."},
    {"role": "user", "content": "kg_summary: ...\nhistory: ...\nintent: explore\nplayer_input: go north\nemotion: curious"},
    {"role": "assistant", "content": "You step towards the north, the cold wind bites your skin..."}
  ]
}
```

### 4.5 使用方法
```bash
# 默认：300 story + 300 option = 600 条
python scripts/generate_dataset.py

# 自定义数量
python scripts/generate_dataset.py --story 500 --option 500

# 只生成 story
python scripts/generate_dataset.py --skip-option --story 400

# 只生成 options
python scripts/generate_dataset.py --skip-story --option 400
```

### 4.6 手动发给 LLM 生成数据
如需手动通过 LLM 聊天界面生成数据，`docs/prompts/` 目录下提供了可直接复制粘贴的 prompt 文件：
- `docs/prompts/story_opening.md` — 故事开场生成 prompt + 上下文素材
- `docs/prompts/story_continuation.md` — 故事续写 prompt + 上下文素材
- `docs/prompts/option_generation.md` — 选项生成 prompt + 上下文素材

每个文件包含：Meta-Prompt（指令）、System Prompt、User Prompt 模板、上下文数据池、以及一个完整的示例。直接复制文件内容发送给任意 LLM 即可批量生成 ChatML 格式的训练数据。

---

## 5. 微调技术方案：ms-swift + LoRA

### 5.1 微调框架
使用阿里开源的 `ms-swift`，它对 Llama3.2 系列模型提供了原生支持，集成了训练、合并、导出、部署全流程。

### 5.2 微调参数
*   **模型**：`Llama-3.2-3B-Instruct`（需要先在 Hugging Face 同意开源协议）
*   **方法**：LoRA (Low-Rank Adaptation) 低秩微调，高效且节省资源
*   **配置参考**：
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

### 5.3 SLURM 批处理脚本 (`train_llama.slurm`)
```bash
#!/bin/bash
#SBATCH --job-name=swift_llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load anaconda3
module load cuda/12.1
conda activate swift_env

swift sft \
    --model_type llama3_2-3b-instruct \
    --dataset /path/to/your/nlg_dataset.jsonl \
    --sft_type lora \
    --output_dir output/nlg_model \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --batch_size 4
```

提交方式：`sbatch train_llama.slurm`

---

## 6. 模型导出与量化

训练完成后（输出 LoRA Adapter 权重），在 HPC 上合并权重并导出为 GGUF 格式（适配本地 CPU 推理）。

### 6.1 导出命令
```bash
# 步骤 1：合并 LoRA 权重
swift export \
    --model_type llama3_2-3b-instruct \
    --adapters output/nlg_model/v0-xxx/checkpoint-xxx \
    --merge_lora true

# 步骤 2：导出为 GGUF 格式 (Q8_0 量化，8-bit)
swift export \
    --model_type llama3_2-3b-instruct \
    --model_id_or_path <合并后的模型目录> \
    --to_gguf true \
    --quant_bits 8 \
    --quant_method q8_0
```

### 6.2 下载
将生成的 `.gguf` 文件（如 `llama3.2-3b-instruct-q8_0.gguf`）通过 SFTP 下载到本地电脑。

---

## 7. 本地部署

使用 **Ollama** 在本地 AMD R7 纯 CPU 环境上高效运行 GGUF 模型。

### 7.1 安装 Ollama
从 [ollama.com](https://ollama.com/) 下载并安装。

### 7.2 导入自定义模型
1.  创建 `Modelfile` 文件：
    ```text
    FROM ./llama3.2-3b-instruct-q8_0.gguf
    ```
2.  终端执行导入命令：
    ```bash
    ollama create storyweaver-llama3 -f Modelfile
    ```

### 7.3 启动服务
Ollama 默认后台运行，自动提供 **OpenAI 兼容 API**：
*   地址：`http://localhost:11434/v1`
*   模型名称：`storyweaver-llama3`
*   API Key：随便填（如 `ollama`），本地不校验

---

## 8. 项目对接与 UI 切换

### 8.1 .env 配置（静态切换）
将本地模型信息写入 `.env` 即可接入：
```env
# 本地模型 (Ollama)
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_MODEL="storyweaver-llama3"
OPENAI_API_KEY="ollama"
```

### 8.2 UI 动态切换（计划）
在 `app.py` 的 Streamlit 侧边栏添加下拉菜单：
*   选项 1: `Cloud API (mimo-v2-flash)`
*   选项 2: `Local Model (storyweaver-llama3)`

用户选择后，动态修改 `config.settings` 并调用 `api_client.py` 中新增的 `reload_client()` 方法刷新底层 OpenAI 客户端实例。

---

## 9. 后续待办事项

| 序号 | 任务 | 状态 |
|------|------|------|
| 1 | 确认 Hugging Face 账号并同意 Llama 3.2 开源协议 | 待确认 |
| 2 | 编写 `scripts/generate_dataset.py` 自动生成训练数据 | 已完成（Prompt-based 版本） |
| 3 | 上传脚本与数据到 PolyU HPC | 待执行 |
| 4 | 编写 `train_llama.slurm` 并提交训练任务 | 待执行 |
| 5 | 合并权重并导出 GGUF 模型 | 待执行 |
| 6 | 本地 Ollama 部署与测试 | 待执行 |
| 7 | 修改 `app.py` 和 `api_client.py` 实现 UI 动态切换 | 待开发 |
| 8 | 评估本地模型与云端 API 的性能对比 | 待执行 |
