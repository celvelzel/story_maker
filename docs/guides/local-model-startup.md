# 本地模型推理启动指南

本文档介绍如何在本地启动模型推理服务并运行 StoryWeaver 应用。

项目支持两种本地推理后端：
- **方案 A: KoboldCpp + Vulkan（推荐 AMD GPU）** — 使用 Vulkan 加速，适合 AMD 显卡
- **方案 B: llama.cpp（纯 CPU）** — 无需 GPU，适合所有平台

---

## 方案 A: KoboldCpp + Vulkan（AMD GPU 加速）

适用于 AMD Radeon 780M 等集成显卡，通过 Vulkan 后端加速推理。

### 前置条件

- Windows 操作系统
- Vulkan 运行时已安装（`vulkaninfo --summary` 可正常输出）
- 已下载 KoboldCpp nocuda 版本至 `C:\Tools\KoboldCpp\`

### 验证 Vulkan 支持

```bash
vulkaninfo --summary
```

确认输出包含：
```
deviceName         = AMD Radeon 780M Graphics
apiVersion         = 1.3.xxx
driverID           = DRIVER_ID_AMD_PROPRIETARY
```

### 一键启动

```bash
scripts\start_local_vulkan.bat
```

该脚本自动完成：
1. 检测并清理占用端口的进程
2. 启动 KoboldCpp（Vulkan 后端）
3. 等待 LLM API 就绪
4. 启动 Streamlit 应用

### 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| **StoryWeaver 应用** | http://127.0.0.1:7860 | Streamlit 前端界面 |
| **KoboldCpp API** | http://127.0.0.1:5001/v1 | OpenAI 兼容 API |

### 配置参数

脚本内可调参数（`start_local_vulkan.bat` 顶部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `KOBOLDCPP` | `C:\Tools\KoboldCpp\koboldcpp.exe` | KoboldCpp 可执行文件路径 |
| `MODEL_PATH` | `models\qwen-gguf\qwen3-4b-q4_k_m.gguf` | GGUF 量化模型 |
| `GPULAYERS` | `99` | GPU 加速层数（全部层） |
| `CONTEXT_SIZE` | `2048` | 上下文长度 |
| `THREADS` | `8` | CPU 线程数 |
| `LLM_PORT` | `5001` | LLM API 端口 |
| `APP_PORT` | `7860` | Streamlit 端口 |

### .env 配置

`.env` 文件应包含：

```bash
# ===== 方案 A：KoboldCpp + Vulkan 加速 =====
OPENAI_BASE_URL=http://127.0.0.1:5001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=koboldcpp/qwen3-4b-q4_k_m
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
```

### 性能参考（AMD 780M）

| 操作 | 耗时 |
|------|------|
| 故事生成 (178 tokens) | ~14s |
| 知识图谱提取 (828 tokens) | ~61s |
| 选项生成 (198 tokens) | ~15s |

---

## 方案 B: llama.cpp 纯 CPU 推理

无需 GPU，适合所有平台，但推理速度较慢。

### 启动方式

```bash
# 步骤 1: 启动本地 llama.cpp 服务器（开启新终端窗口）
scripts\start_llama_server.bat

# 步骤 2: 启动 StoryWeaver 应用（另开新终端窗口）
scripts\start_project_prod.bat
```

### 访问地址

- **StoryWeaver 应用**: http://127.0.0.1:7860
- **llama.cpp API 端点**: http://127.0.0.1:8081/v1

### 配置参数

脚本内可调参数（`start_llama_server.bat` 内部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_BIN` | `llama.cpp-bin\llama-server.exe` | llama-server 可执行文件 |
| `MODEL_PATH` | `models\qwen-gguf\qwen3-4b-q4_k_m.gguf` | GGUF 量化模型 |
| `PORT` | `8081` | API 服务端口 |
| `CONTEXT_SIZE` | `2048` | 上下文长度 |
| `BATCH_SIZE` | `512` | 批处理大小 |
| `THREADS` | `4` | CPU 线程数 |

### .env 配置

```bash
# ===== 方案 B：llama.cpp 本地 CPU 推理 =====
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=qwen3-4b
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
```

---

## 依赖文件

确保以下文件存在：

1. **推理引擎**（二选一）:
   - KoboldCpp: `C:\Tools\KoboldCpp\koboldcpp.exe`
   - llama-server: `llama.cpp-bin/llama-server.exe`
2. **量化模型**: `models/qwen-gguf/qwen3-4b-q4_k_m.gguf`
3. **环境配置**: `.env`

### 检查模型文件

```bash
dir models\qwen-gguf\
```

应看到：
- `qwen3-4b-q4_k_m.gguf` — Q4 量化模型（推荐，2.4GB）
- `qwen3-4b-f16.gguf` — FP16 模型（7.5GB，需要更多内存）

---

## 演示模式日志

启动应用后，LLM 请求会显示：

```
==================================================
[LLM] RECEIVED request
==================================================
  [system]: You are an expert interactive-fiction narrator...
  [user]: Create the opening scene...
==================================================
[LLM] PROCESSING (model=koboldcpp/qwen3-4b-q4_k_m, temp=0.8, max_tokens=512)
[LLM] DONE (elapsed: 14.0s, input: 178 tokens, output: 178 tokens)
```

日志字段说明：
- **RECEIVED request**: 收到聊天请求，显示消息角色和内容预览
- **PROCESSING**: 模型正在生成，包含配置参数
- **DONE**: 生成完成，显示耗时和 token 使用量

---

## 故障排查

### 端口被占用

```bash
# Windows: 检查端口占用
netstat -ano | findstr :5001
netstat -ano | findstr :7860
netstat -ano | findstr :8081

# 终止占用进程
taskkill /PID <PID> /F
```

### Vulkan 不可用

```bash
# 检查 Vulkan 支持
vulkaninfo --summary

# 若命令不存在，安装 Vulkan Runtime:
# https://vulkan.lunarg.com/sdk/home#windows
```

### 模型文件找不到

确保模型文件在 `models/qwen-gguf/` 目录下。

### 连接失败

1. 检查推理服务是否运行
2. 确认端口可访问：
   - KoboldCpp: `curl http://127.0.0.1:5001/v1/models`
   - llama.cpp: `curl http://127.0.0.1:8081/v1/models`
3. 查看 `.env` 配置的 `OPENAI_BASE_URL` 是否正确

### HuggingFace 网络超时

若启动时出现 HuggingFace 重试日志（`fastcoref` 模块），这是因为网络不可达。系统会自动回退到规则模式，不影响功能。详见 [fastcoref HuggingFace 超时修复文档](../fixes/fastcoref-huggingface-timeout-fix.md)。

---

## 性能提示

| 后端 | 推荐场景 | 速度 |
|------|---------|------|
| **KoboldCpp + Vulkan** | AMD GPU 用户 | ⭐⭐⭐ 快 |
| **llama.cpp CPU** | 无 GPU / 通用 | ⭐ 慢 |

- **量化模型**: 使用 Q4_K_M 量化，内存占用约 2.4GB
- **上下文长度**: 根据故事长度需求调整 `--ctx-size` / `--contextsize`
- **GPU 层数**: AMD 集成显卡建议使用全部层 (`--gpulayers 99`)

---

## 相关文档

- [KoboldCpp 官方文档](https://github.com/LostRuins/koboldcpp/wiki)
- [本地模型推理集成报告](../reports/本地模型推理集成_2026-03-27_21-07.md)
- [CPU 推理优化指南](../../CPU_INFERENCE.md)
- [vLLM 集成指南](../../VLLM_INTEGRATION.md)
