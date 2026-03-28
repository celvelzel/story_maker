# Windows 高可用部署指南

## 1. 推理后端选择

Windows 部署支持两种本地推理后端：

| 后端 | 硬件要求 | 启动脚本 | 推荐场景 |
|------|---------|---------|---------|
| **KoboldCpp + Vulkan** | AMD GPU | `start_local_vulkan.bat` | ⭐ AMD 显卡用户推荐 |
| **llama.cpp CPU** | 无要求 | `start_llama_server.bat` + `start_project_prod.bat` | 通用，无需 GPU |

---

## 2. 方案 A: KoboldCpp + Vulkan（AMD GPU 加速）

### 2.1 前置条件

1. AMD 显卡（如 Radeon 780M / RX 6000/7000 系列）
2. Vulkan 运行时已安装（显卡驱动通常自带）
3. KoboldCpp 已下载至 `C:\Tools\KoboldCpp\koboldcpp.exe`

### 2.2 验证 Vulkan 支持

```cmd
vulkaninfo --summary
```

确认输出包含 `AMD Radeon` 字样和 `apiVersion >= 1.3`。

### 2.3 一键启动

```cmd
scripts\start_local_vulkan.bat
```

### 2.4 启动流程说明

脚本自动完成：
1. 检测并清理占用端口 5001 的进程
2. 验证 KoboldCpp 可执行文件和模型文件
3. 启动 KoboldCpp（Vulkan 后端）
4. 等待 LLM API 就绪（最多 120 秒）
5. 启动 Streamlit 应用

### 2.5 访问地址

| 服务 | 地址 |
|------|------|
| StoryWeaver 应用 | http://127.0.0.1:7860 |
| LLM API | http://127.0.0.1:5001/v1 |

### 2.6 配置文件 (.env)

```env
OPENAI_BASE_URL=http://127.0.0.1:5001/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=koboldcpp/qwen3-4b-q4_k_m
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

### 2.7 日志

- KoboldCpp 日志：`logs/koboldcpp_stdout.log`, `logs/koboldcpp_stderr.log`
- 脚本日志：`logs/vulkan_deploy_*.log`

---

## 3. 方案 B: llama.cpp 纯 CPU 推理

适用于无 GPU 或非 AMD 显卡的场景。

### 3.1 启动方式

需要两个终端窗口：

**终端 1 - 启动 LLM 服务：**
```cmd
scripts\start_llama_server.bat
```

**终端 2 - 启动应用：**
```cmd
scripts\start_project_prod.bat
```

### 3.2 访问地址

| 服务 | 地址 |
|------|------|
| StoryWeaver 应用 | http://127.0.0.1:7860 |
| LLM API | http://127.0.0.1:8081/v1 |

### 3.3 配置文件 (.env)

```env
OPENAI_BASE_URL=http://127.0.0.1:8081/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=qwen3-4b
OPENAI_TIMEOUT_CONNECT=30.0
OPENAI_TIMEOUT_READ=180.0
OPENAI_MAX_TOKENS=512
OPENAI_TEMPERATURE=0.8
```

---

## 4. 脚本说明

| 脚本 | 用途 |
|------|------|
| `start_local_vulkan.bat` | 一键启动 KoboldCpp + Streamlit（AMD GPU） |
| `start_llama_server.bat` | 启动 llama.cpp 服务器（CPU） |
| `start_project_prod.bat` | 启动 Streamlit 应用 |

---

## 5. 故障排查

### 端口被占用

```cmd
netstat -ano | findstr :5001
netstat -ano | findstr :7860
netstat -ano | findstr :8081

taskkill /PID <PID> /F
```

### Vulkan 不可用

```cmd
vulkaninfo --summary
```

若命令不存在，安装 Vulkan Runtime：
https://vulkan.lunarg.com/sdk/home#windows

### 模型文件找不到

确保模型文件在 `models\qwen-gguf\qwen3-4b-q4_k_m.gguf`。

### LLM 服务未响应

检查对应的 LLM 服务终端是否正常运行：
- KoboldCpp: `curl http://127.0.0.1:5001/v1/models`
- llama.cpp: `curl http://127.0.0.1:8081/v1/models`

### HuggingFace 网络超时

启动时若出现 fastcoref 重试日志，这是因为网络不可达。系统会自动回退到规则模式，不影响功能。

---

## 6. 性能参考

| 后端 | 硬件 | 故事生成 (~180 tokens) | 选项生成 (~200 tokens) |
|------|------|----------------------|---------------------|
| KoboldCpp Vulkan | AMD 780M | ~14s | ~15s |
| llama.cpp CPU | 通用 | ~30-60s | ~30-45s |

---

## 7. 回滚建议

1. 保留上一个稳定版本代码包
2. 回滚后重新执行启动脚本
3. 若模型工件有变更，同步回滚 `models/intent_classifier`
