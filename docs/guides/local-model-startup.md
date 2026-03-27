# 本地模型推理启动指南

本文档介绍如何在本地启动 llama.cpp 服务器并运行 StoryWeaver 应用。

## 1. 快速启动

### Windows

```bash
# 步骤 1: 启动本地 llama.cpp 服务器（开启新终端窗口）
scripts\start_llama_server.bat

# 步骤 2: 启动 StoryWeaver 应用（另开新终端窗口）
scripts\start_project_prod.bat
```

### 访问地址

- **StoryWeaver 应用**: http://127.0.0.1:7860
- **llama.cpp API 端点**: http://127.0.0.1:8081/v1/chat/completions

## 2. 启动脚本说明

### start_llama_server.bat

启动本地 llama.cpp 服务器，提供 OpenAI 兼容 API。

**配置参数**（可修改脚本内变量）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_BIN` | `llama.cpp-bin\llama-server.exe` | llama-server 可执行文件 |
| `MODEL_PATH` | `models\qwen-gguf\qwen3-4b-q4_k_m.gguf` | GGUF 量化模型 |
| `PORT` | `8081` | API 服务端口 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `CONTEXT_SIZE` | `2048` | 上下文长度 |
| `BATCH_SIZE` | `512` | 批处理大小 |
| `THREADS` | `4` | CPU 线程数 |

**启动后显示**：

```
=============================================
  llama.cpp Local API Server
=============================================

[INFO] Model:    models\qwen-gguf\qwen3-4b-q4_k_m.gguf
[INFO] Server:   http://127.0.0.1:8081
[INFO] Context:  2048 tokens
[INFO] Threads:  4

OpenAI-compatible endpoints:
  Chat:    http://127.0.0.1:8081/v1/chat/completions
  Models:  http://127.0.0.1:8081/v1/models
```

### start_project_prod.bat

启动 StoryWeaver Streamlit 应用。

**功能特性**：
- 自动检测并处理占用端口的进程
- 自动创建虚拟环境并安装依赖
- 日志输出到 `logs/` 目录

## 3. 依赖文件

确保以下文件存在：

1. **llama-server**: `llama.cpp-bin/llama-server.exe`
2. **量化模型**: `models/qwen-gguf/qwen3-4b-q4_k_m.gguf`
3. **环境配置**: `.env`（需配置本地后端选项）

### 检查模型文件

```bash
dir models\qwen-gguf\
```

应看到：
- `qwen3-4b-q4_k_m.gguf` - Q4 量化模型（推荐，2.4GB）
- `qwen3-4b-f16.gguf` - FP16 模型（7.5GB，需要更多内存）

## 4. 配置文件 (.env)

项目默认配置为本地 llama.cpp 后端。`.env` 文件示例：

```bash
# ===== 方案 C：llama.cpp 本地 CPU 推理（默认激活）=====
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_MODEL=qwen3-4b
OPENAI_API_KEY=local
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=512
```

如需切换到远程 API 或 vLLM，参考 `docs/reports/本地模型推理集成_2026-03-27_21-07.md`。

## 5. 演示模式日志

启动应用后，当 LLM 请求发出时，llama.cpp 服务器终端会显示：

```
==================================================
[LLM] 📥 收到请求
==================================================
  [system]: You are a helpful assistant...
  [user]: Create a story about...
==================================================
[LLM] 🔄 正在处理 (model=qwen3-4b, temp=0.7, max_tokens=512)
[LLM] ✅ 完成! (耗时: 15.2s, 输入: 128 tokens, 输出: 256 tokens)
```

日志显示：
- **📥 收到请求**: 收到聊天请求，显示消息角色和内容预览
- **🔄 正在处理**: 模型正在生成，包含配置参数
- **✅ 完成**: 生成完成，显示耗时和 token 使用量

## 6. 故障排查

### 端口被占用

```bash
# Windows: 检查端口占用
netstat -ano | findstr :8081
netstat -ano | findstr :7860

# 终止占用进程
taskkill /PID <PID> /F
```

### 模型文件找不到

确保模型文件在 `models/qwen-gguf/` 目录下。参考文档 `docs/guides/../CPU_INFERENCE.md` 进行模型转换。

### 连接失败

1. 检查 llama.cpp 服务器是否运行（终端窗口是否打开）
2. 确认端口 `8081` 可访问：`curl http://127.0.0.1:8081/v1/models`
3. 查看 `.env` 配置的 `OPENAI_BASE_URL` 是否正确

## 7. 性能提示

- **量化模型**: 使用 Q4_K_M 量化，内存占用约 2.4GB
- **CPU 线程**: 根据 CPU 核心数调整 `--threads` 参数
- **上下文长度**: 根据故事长度需求调整 `--ctx-size`

## 8. 相关文档

- [本地模型推理集成报告](../reports/本地模型推理集成_2026-03-27_21-07.md)
- [CPU 推理优化指南](../CPU_INFERENCE.md)
- [vLLM 集成指南](../VLLM_INTEGRATION.md)