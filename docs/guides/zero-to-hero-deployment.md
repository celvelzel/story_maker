# StoryWeaver 从零开始全平台部署指南 (Zero to Hero)

本文档旨在为开发人员（或 AI Agent）提供在**全新的 Windows 或 macOS 机器**上，从零开始部署 StoryWeaver 项目的完整指导，支持 **CPU / CUDA / Apple Silicon (Metal)** 三种推理模式。

---

## 1. 架构与依赖总览

StoryWeaver 是一个多模态、大语言模型驱动的文字冒险游戏。它的核心依赖包括：
- **前端与中间件**：Python 3.10+ (推荐), Streamlit, NetworkX, PyVis 等。
- **本地 NLU (自然语言理解)**：PyTorch, Transformers (DistilBERT), spaCy, fastcoref。
- **本地 NLG (自然语言生成)**：`llama.cpp`，作为一个独立的 OpenAI 兼容 API 服务器运行。

### 1.1 推理后端选择矩阵

| 硬件配置 | 推荐模式 | 加速方式 | 预期速度 |
|---------|---------|---------|---------|
| **AMD GPU (集成/独立)** | KoboldCpp Vulkan ⭐推荐 | Vulkan | ~10-30 tokens/s |
| **仅 CPU** | llama.cpp CPU | - | ~5-15 tokens/s |
| **NVIDIA GPU** | llama.cpp CUDA | cuBLAS / cuDNN | ~30-80 tokens/s |
| **Apple Silicon (M1/M2/M3)** | llama.cpp Metal | Apple Metal | ~20-50 tokens/s |
| **无本地硬件** | 远程 API | - | 取决于网络 |

---

## 2. 准备工作

### 2.1 环境要求
- **操作系统**：Windows 10/11 或 macOS (Apple Silicon 推荐)。
- **Python**：版本 `3.10` - `3.12`。
- **C/C++ 编译器** (仅从源码编译 llama.cpp 时需要)：
  - **Windows**: Visual Studio 2022 (包含 C++ 桌面开发工作负载) 或 MinGW。
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)。

### 2.2 下载模型文件
该项目本地推理推荐使用 Qwen 模型的 GGUF 量化版本。
1. 创建目录 `models/qwen-gguf/`。
2. 下载模型文件 `qwen3-4b-q4_k_m.gguf` （约 2.4GB）并放置在该目录下。
   > 注意：如果没有这个模型，可以通过 HuggingFace 下载转换，参考文档 `docs/CPU_INFERENCE.md`。

---

## 3. 推理后端配置

根据硬件选择对应的推理后端。

### 3.1 方案 A: KoboldCpp + Vulkan（AMD GPU 推荐）

适用于 AMD Radeon 780M / RX 6000/7000 系列等显卡，通过 Vulkan 后端加速推理。

#### 前置条件

- Windows 操作系统
- Vulkan 运行时已安装（显卡驱动通常自带）

#### 验证 Vulkan 支持

```cmd
vulkaninfo --summary
```

确认输出包含：
```
deviceName         = AMD Radeon 780M Graphics  (或其他 AMD GPU)
apiVersion         = 1.3.xxx
driverID           = DRIVER_ID_AMD_PROPRIETARY
```

#### 下载 KoboldCpp

1. 前往 https://github.com/LostRuins/koboldcpp/releases
2. 下载 **nocuda** 版本（推荐 AMD 用户使用 Vulkan）：
   - 文件名：`koboldcpp-nocuda.exe`（约 97MB）
3. 创建目录 `C:\Tools\KoboldCpp\`，将下载的文件重命名为 `koboldcpp.exe` 并放入

#### 一键启动

```cmd
scripts\start_local_vulkan.bat
```

该脚本自动完成：
1. 检测并清理占用端口的进程
2. 启动 KoboldCpp（Vulkan 后端，端口 5001）
3. 等待 LLM API 就绪
4. 启动 Streamlit 应用（端口 7860）

#### 访问地址

| 服务 | 地址 |
|------|------|
| StoryWeaver 应用 | http://127.0.0.1:7860 |
| KoboldCpp API | http://127.0.0.1:5001/v1 |

#### 手动启动（调试用）

```cmd
C:\Tools\KoboldCpp\koboldcpp.exe --usevulkan --gpulayers 99 --model models\qwen-gguf\qwen3-4b-q4_k_m.gguf --contextsize 2048 --threads 8 --port 5001
```

#### .env 配置

```env
# ===== KoboldCpp + Vulkan =====
OPENAI_BASE_URL=http://127.0.0.1:5001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=koboldcpp/qwen3-4b-q4_k_m
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

#### 性能参考（AMD 780M）

| 操作 | 耗时 |
|------|------|
| 故事生成 (~180 tokens) | ~14s |
| 知识图谱提取 (~830 tokens) | ~61s |
| 选项生成 (~200 tokens) | ~15s |

---

### 3.2 方案 B: llama.cpp（通用 CPU / NVIDIA / Apple Silicon）

### 3.1 获取 llama-server（根据硬件选择）

#### 方案 A: 预编译二进制（通用）

1. 前往 https://github.com/ggerganov/llama.cpp/releases 下载对应平台的预编译版本：
   - **Windows**: 下载 `llama.cpp-bin-win-*.zip`
   - **macOS (Intel)**: 下载 `llama.cpp-bin-macos-intel-*.zip`
   - **macOS (Apple Silicon)**: 下载 `llama.cpp-bin-macos-arm64-*.zip`
2. 解压到项目的 `llama.cpp-bin/` 目录下。确保 `llama.cpp-bin/llama-server.exe`（或 `llama-server`）存在。

#### 方案 B: 源码编译（启用 GPU 加速）(推荐)

**通用编译：**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j
```

**启用 CUDA 加速（NVIDIA GPU）：**
```bash
# 需要安装 CUDA Toolkit
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1 -j
# 编译后自动使用 GPU 加速
```

**启用 Metal 加速（Apple Silicon M1/M2/M3）：**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1 -j
# 注意：macOS 编译 Metal 版本需要 Xcode
```

编译完成后，将 `llama-server` 可执行文件复制到项目对应位置。

### 3.2 启动 llama.cpp API 服务器

服务器脚本会在 `127.0.0.1:8081` 启动服务。

#### 3.2.1 Windows 启动

打开 `cmd` (不要使用 PowerShell，避免中文编码和执行策略问题)：
```cmd
cd \path\to\story_maker
scripts\start_llama_server.bat
```

#### 3.2.2 macOS / Linux 启动

**CPU 模式：**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml
```

**Apple Silicon Metal 加速（推荐）：**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 8 --ngl 99 --chat-template chatml
```
> `-t 8`: 根据 M 系列核心数调整线程  
> `--ngl 99`: 将所有层加载到 GPU（Metal）

#### 3.2.3 CUDA 启动（NVIDIA GPU）

```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 4096 -b 512 -t 8 --ngl 99 --gpu-layers 99 --chat-template chatml
```
> 注意：确保 CUDA 驱动和 cuDNN 已正确安装

**验证服务：**
打开新终端，运行：
```bash
curl -s http://127.0.0.1:8081/v1/models
```
如果有 JSON 返回，说明大模型服务就绪。

---

## 4. 主项目与前端部署

主项目使用虚拟环境和依赖文件管理，内置了高度自动化的部署脚本。

### 4.1 配置文件 (`.env`)

复制 `.env.example` 为 `.env`。根据你使用的推理后端，选择对应的配置：

**方案 A: KoboldCpp + Vulkan（AMD GPU）：**
```env
OPENAI_BASE_URL=http://127.0.0.1:5001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=koboldcpp/qwen3-4b-q4_k_m
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

**方案 B: llama.cpp（CPU / CUDA / Metal）：**
```env
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=qwen3-4b
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

> **GPU 加速提示**：如果使用 CUDA 或 Metal 加速，可以将超时改为更短的值（如 60s），因为 GPU 推理速度通常是 CPU 的 5-10 倍。

### 4.2 启动 Streamlit 前端应用

项目提供了高度自动化的脚本，它会自动创建虚拟环境 (`.venv`)、安装 `requirements.txt` 中的依赖，并启动 Streamlit。

**在 Windows 下：**
在新的 `cmd` 或 PowerShell 窗口运行：
```cmd
scripts\start_project_prod.bat
```

**在 macOS 下：**
```bash
chmod +x scripts/start_project_prod.sh
./scripts/start_project_prod.sh
```

**首次启动注意事项：**
1. **依赖下载较慢**：脚本包含超时重试机制，但 `torch` 等大包可能下载缓慢。如果中途失败，再次运行脚本即可恢复进度。
2. **spaCy 模型**：脚本会自动下载 `en_core_web_sm` 模型。如果因网络问题报错，可进入虚拟环境手动下载：
   ```bash
   .venv/bin/python -m spacy download en_core_web_sm
   ```

启动成功后，浏览器会自动打开 `http://127.0.0.1:7860`。

---

## 5. 备选方案：vLLM（需要 NVIDIA GPU）

如果用户拥有 NVIDIA GPU 且追求极致性能，可以使用 **vLLM** 替代 llama.cpp。vLLM 支持 PagedAttention，推理速度更快。

### 5.1 vLLM 安装与启动

参考文档 `docs/VLLM_INTEGRATION.md`，核心步骤：

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 启动 vLLM 服务器（默认端口 8000）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct-agreeable-gguf \
    --gguf-repo-id Qwen/Qwen2.5-3B-Instruct-GGUF \
    --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --dtype half \
    --port 8000
```

### 5.2 配置 StoryWeaver 使用 vLLM

在 `.env` 中配置：

```env
# vLLM API
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=qwen-local

# vLLM 推理速度较快，可以使用较短超时
OPENAI_TIMEOUT_CONNECT=10.0
OPENAI_TIMEOUT_READ=60.0
```

---

## 6. 联调与 Debug

当所有组件运行后，你可以通过查看日志来监控系统状态：

### 6.1 前端应用日志
- Streamlit 后台会打印运行日志。
- 文件日志保存在 `logs/storyweaver_prod_YYYYMMDD_HHMMSS.log`。

### 6.2 请求交互日志
在你的 `start_project_prod` 终端中，每当与大模型交互时，会打印带有特定格式的交互日志，例如：
```
==================================================
[LLM] RECEIVED request
==================================================
  [system]: You are a helpful assistant...
  [user]: The player steps into the dark room...
==================================================
[LLM] PROCESSING (model=qwen3-4b, temp=0.8, max_tokens=512)
[LLM] DONE (elapsed: 15.2s, input: 128 tokens, output: 256 tokens)
```

### 6.3 常见问题排查

1. **`Connection error` 或请求超时**：
   - 检查 `.env` 中的 `OPENAI_TIMEOUT_READ` 是否设置得足够大 (如 `180.0` 或更大)。
   - 检查 llama.cpp 服务器终端是否正在处理请求，以及是否卡死。

2. **端口被占用 (`7860` 或 `8081`)**：
   - 脚本会自动尝试关闭旧的 Streamlit 进程。
   - 对于 `8081`，可以手动杀死：
     - Windows: `netstat -ano | findstr :8081` 找到 PID，然后 `taskkill /PID <PID> /F`
     - macOS: `lsof -i :8081` 找到 PID，然后 `kill -9 <PID>`

3. **编码报错 (`gbk codec can't encode character`)**：
   - 这是 Windows 控制台环境导致的。项目代码中已移除可能导致冲突的 Emoji，确保代码更新至最新。

4. **NLU 意图分类/实体提取不准确**：
   - NLU 组件依赖 `models/intent_classifier/` 下的模型。如果目录不存在，系统将退回“关键字匹配”模式。你可以通过 `python training/train_intent.py` 重新训练。