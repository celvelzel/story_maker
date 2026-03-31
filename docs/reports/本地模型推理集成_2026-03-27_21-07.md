# 本地模型推理集成报告 (llama.cpp)

**日期：** 2026年03月27日 21:07  
**目标：** 实现基于 llama.cpp 的本地 CPU 推理支持，集成 Qwen3-4B 量化模型。

---

## 1. 变更摘要

新增对 llama.cpp 推理后端的支持。系统现在可以在本地 CPU 上运行 Qwen3-4B GGUF 量化模型，完全摆脱对远程 API 或 GPU 的依赖，同时通过 OpenAI 兼容接口保持与现有架构的无缝集成。

## 2. 修改详情

### 2.1 新增文件

| 文件路径 | 类型 | 说明 |
|---------|---------|------|
| `scripts/start_llama_server.bat` | 脚本 | Windows 平台 llama.cpp 服务器一键启动脚本 |
| `scripts/start_llama_server.sh` | 脚本 | Linux/macOS 平台 llama.cpp 服务器启动脚本 |
| `.env.llama` | 配置 | 本地模型专用配置模板 |

### 2.2 核心配置更新 (`.env`)

更新了后端推理方案的分级支持：
- **方案 A (远程)**: OpenAI / Mimo API (低延迟，需联网)
- **方案 B (本地 GPU)**: vLLM (高性能，需 NVIDIA GPU)
- **方案 C (本地 CPU)**: llama.cpp (高兼容性，支持纯 CPU 运行，**当前默认激活**)

### 2.3 功能实现

- **一键启动脚本**:
  - 自动检测模型文件 (`models/*.gguf`) 和服务器二进制文件。
  - 支持自定义端口 (默认 8081)、线程数、上下文长度和批处理大小。
  - 启动 OpenAI 兼容的 REST API。
- **UI 适配计划**:
  - 侧边栏新增后端选择器。
  - 实现动态 LLM 客户端切换与服务器状态实时检测。

## 3. 技术规范

- **模型转换**: HuggingFace (PyTorch) → GGUF (FP16) → Quantized (Q4_K_M)。
- **压缩效率**: 模型大小从 7.5GB (FP16) 压缩至 2.4GB (Q4_K_M)，内存占用降低约 68%。
- **后端服务**: 基于 llama-server b5200 版本。
- **接口标准**: 
  - Chat completions: `http://127.0.0.1:8081/v1/chat/completions`
  - Model info: `http://127.0.0.1:8081/v1/models`

## 4. 测试与验证

- **配置加载**: ✅ 成功读取 `OPENAI_BASE_URL` 与 `OPENAI_MODEL`。
- **客户端兼容性**: ✅ `api_client.py` 成功调用本地 8081 端口，响应格式符合 OpenAI 规范。
- **推理稳定性**: ✅ 在 16GB 内存机器上运行稳定，支持 4096 上下文。

## 5. 使用指南

1. **启动推理服务器**:
   ```bash
   # Windows
   scripts\start_llama_server.bat
   # Linux/macOS
   bash scripts/start_llama_server.sh
   ```
2. **运行 StoryWeaver**:
   ```bash
   python -m streamlit run app.py
   ```

## 6. 后续计划

- 在 Streamlit UI 中添加模型热切换功能。
- 引入更多量化档位 (Q2_K, Q5_K, Q8_0) 以适配不同内存配置。
- 优化 CPU 多线程调度，提升长文本生成速度。
