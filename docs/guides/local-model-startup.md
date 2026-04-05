# 本地模型推理启动指南

本文档描述如何在本地启动 llama.cpp 服务器并运行 StoryWeaver 应用。

> **最后更新**：2026-04-01

## 前置条件

- [llama.cpp](https://github.com/ggerganov/llama.cpp) 二进制文件（放置在 `llama.cpp-bin/` 目录中）
- GGUF 模型文件位于 `models/qwen-gguf/` 目录中
- Windows/macOS/Linux 环境，Python 3.10+

## 1. 快速开始

### Windows 系统

```powershell
# 步骤 1：启动本地 llama.cpp 服务器（打开新的终端窗口）
.\scripts\start_llama_server.bat

# 步骤 2：启动 StoryWeaver 应用（再打开一个新的终端窗口）
.\scripts\start_project_prod.bat
```

### macOS / Linux 系统

```bash
# 步骤 1：启动本地 llama.cpp 服务器（打开新的终端窗口）
# 方案 A：使用批处理脚本包装器（如果可用）
# 方案 B：直接运行 llama-server：
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml

# 步骤 2：启动 StoryWeaver 应用（打开新的终端窗口）
./scripts/start_project_prod.sh
```

### 访问地址

- **StoryWeaver 应用**：[http://127.0.0.1:7860](http://127.0.0.1:7860)
- **llama.cpp API 端点**：[http://127.0.0.1:8081/v1/chat/completions](http://127.0.0.1:8081/v1/chat/completions)

## 2. 启动脚本详情

### start_llama_server.bat（Windows 系统）

启动本地 llama.cpp 服务器，提供 OpenAI 兼容的 API。

**配置参数**（可在脚本内调整）：

| 参数 | 默认值 | 描述 |
|-----------|---------------|-------------|
| `LLAMA_BIN` | `llama.cpp-bin\llama-server.exe` | llama-server 可执行文件路径 |
| `MODEL_PATH` | `models\qwen-gguf\qwen3-4b-q4_k_m.gguf` | GGUF 量化模型路径 |
| `PORT` | `8081` | API 服务端口 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `CONTEXT_SIZE` | `2048` | 上下文长度（token 数） |
| `BATCH_SIZE` | `512` | 批次大小 |
| `THREADS` | `4` | 使用的 CPU 线程数 |

**预期控制台输出**：

```text
=========================================
  llama.cpp 本地 API 服务器
=========================================

[INFO] 模型：    models\qwen-gguf\qwen3-4b-q4_k_m.gguf
[INFO] 服务器：  http://127.0.0.1:8081
[INFO] 上下文：  2048 token
[INFO] 线程数：  4

OpenAI 兼容端点：
  聊天：    http://127.0.0.1:8081/v1/chat/completions
  模型列表：http://127.0.0.1:8081/v1/models
```

### macOS/Linux：直接运行 llama-server 命令

项目没有提供 `start_llama_server.sh` 脚本，直接运行 `llama-server`：

**CPU 模式：**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml
```

**Apple Silicon Metal 加速（推荐）：**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 8 --ngl 99 --chat-template chatml
```

**NVIDIA CUDA 加速：**
```bash
./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 4096 -b 512 -t 8 --ngl 99 --chat-template chatml
```

### start_project_prod（bat/sh 脚本）

以生产模式启动 StoryWeaver Streamlit 应用。

**功能**：
- 自动检测并处理占用目标端口的进程
- 创建虚拟环境并安装缺失的依赖
- 将日志重定向到 `logs/` 目录

## 3. 必需文件

确保以下文件存在：

1. **llama-server**：`llama.cpp-bin/llama-server`（Windows 为 `.exe`）
2. **量化模型**：`models/qwen-gguf/qwen3-4b-q4_k_m.gguf`
3. **配置文件**：`.env`（配置为本地后端）

### 验证模型文件

```bash
ls -lh models/qwen-gguf/
```

预期输出：
- `qwen3-4b-q4_k_m.gguf` — Q4 量化模型（约 2.4GB，推荐）

## 4. 环境配置（.env）

项目通过 `config.py` 的 `NLG_MODE` 支持三种 NLG 模式：`api`、`local`、`hybrid`。

**本地 llama.cpp 后端配置**：

```ini
# llama.cpp 本地推理
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_MODEL=qwen3-4b
OPENAI_API_KEY=local

# 超时配置（根据硬件调整）
# CPU 推理：需要较长超时
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0

# GPU（CUDA/Metal）推理：可以使用较短超时
# OPENAI_TIMEOUT_CONNECT=10.0
# OPENAI_TIMEOUT_READ=60.0

OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

切换到远程 API 时，相应更新 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。远程 API 模板请参考 `config/.env.example`。

## 5. 实时日志

当应用发送 LLM 请求时，llama.cpp 服务器终端会显示请求处理信息。

## 6. 故障排查

### 端口已被占用

**Windows 系统**：
```powershell
netstat -ano | findstr :8081
taskkill /PID <PID> /F
```

**macOS/Linux 系统**：
```bash
lsof -i :8081
kill -9 <PID>
```

### 模型未找到

确保模型文件位于 `models/qwen-gguf/` 目录中。模型下载说明请参考 [从零开始部署指南](zero-to-hero-deployment.md)。

### 连接失败

1. 检查 llama.cpp 服务器是否正在运行
2. 验证端口 `8081` 是否可访问：`curl http://127.0.0.1:8081/v1/models`
3. 确认 `.env` 中的 `OPENAI_BASE_URL` 与服务器地址匹配

## 7. 性能优化建议

- **量化**：使用 `Q4_K_M` 在速度和质量之间取得良好平衡（约占用 2.4GB 内存）
- **线程数**：将 `--threads` 参数设置为物理 CPU 核心数
- **上下文大小**：如果计划进行长时间故事会话，增加 `--ctx-size`
- **Metal/CUDA**：使用 `--ngl 99` 将所有层卸载到 GPU，可显著提升速度

## 8. 相关文档

- [从零开始部署指南](zero-to-hero-deployment.md)
- [CPU 推理（已废弃）](CPU_INFERENCE.md)

---
*最后更新：2026-04-01*
