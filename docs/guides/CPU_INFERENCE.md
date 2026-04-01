# 纯 CPU 推理 — 已废弃

> **⚠️ 已废弃（2026-04-01）**：本文档仅供历史参考。本文档原始版本中引用的 vLLM CPU 推理脚本、配置文件及相关基础设施**已不存在**于本仓库中。
>
> **请改用 llama.cpp。** 请参考 [本地模型启动指南](local-model-startup.md) 和 [从零开始部署指南](zero-to-hero-deployment.md)。

---

## 已移除的内容

本文档原始版本中描述的以下文件和配置已被移除：

| 文件 | 状态 |
|------|--------|
| `scripts/start_vllm_server_cpu.sh` | ❌ 已移除 |
| `scripts/start_vllm_server.sh` | ❌ 已移除 |
| `.env.vllm.cpu` | ❌ 已移除 |
| `.env.vllm` / `.env.vllm.example` | ❌ 已移除 |
| `VLLM_INTEGRATION.md` | ❌ 已移除 |

## 为什么 llama.cpp 取代了 vLLM 用于 CPU 推理

| 因素 | vLLM（旧版） | llama.cpp（当前） |
|--------|-----------|---------------------|
| CPU 支持 | 实验性，设置复杂 | 原生支持，开箱即用 |
| 部署复杂度 | 高（Python 依赖、量化流水线） | 低（单个二进制文件） |
| 内存占用 | 较高 | 较低 |
| 跨平台 | 主要面向 Linux | Windows/macOS/Linux |
| GGUF 支持 | 有限 | 原生支持 |
| Metal（Apple Silicon） | 不支持 | 支持 |
| CUDA | 支持 | 支持 |

## 迁移路径

如果你之前参考的是本文档，请切换到：

1. **[从零开始部署](zero-to-hero-deployment.md)** — 从零开始完整设置
2. **[本地模型启动](local-model-startup.md)** — 快速启动 llama.cpp 服务器

### 快速迁移步骤

```bash
# 1. 下载 llama.cpp 二进制文件
#    下载地址：https://github.com/ggerganov/llama.cpp/releases
#    解压到 llama.cpp-bin/ 目录

# 2. 放置 GGUF 模型文件
mkdir -p models/qwen-gguf/
# 将 qwen3-4b-q4_k_m.gguf 放到 models/qwen-gguf/ 目录中

# 3. 启动 llama.cpp 服务器
#    Windows 系统：scripts\start_llama_server.bat
#    macOS/Linux 系统：./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048

# 4. 配置 .env 文件
#    OPENAI_BASE_URL=http://127.0.0.1:8081/v1
#    OPENAI_MODEL=qwen3-4b
#    OPENAI_API_KEY=local
```

---

*本文档已归档。最后有意义更新：2026-03-31。废弃日期：2026-04-01。*
