# CPU-Only 推理优化指南

## 概述

本文档说明如何在**无独显（纯CPU）** 的硬件上运行 Qwen 2.5 3B 模型，使用 vLLM + INT4 量化实现高效推理。

### 目标硬件配置
- **CPU**: AMD Ryzen 7 H255W（8核）或同等规格
- **内存**: 32GB RAM
- **独显**: 无
- **存储**: 至少 5GB（用于模型和量化缓存）

---

## 为什么需要 INT4 量化？

### 问题
- **原始模型大小**: Qwen 2.5 3B 原始精度（FP32）≈ 12GB
- **内存限制**: 32GB 内存中需要预留给系统（~2-3GB）和推理过程（激活、缓存等）
- **无法适配**: 在 CPU 上直接加载 FP32 模型会导致 OOM（内存溢出）

### INT4 量化的好处

| 指标 | FP32 | INT4 |
|------|------|------|
| 模型大小 | ~12GB | ~2.5GB |
| 内存占用 | 12-14GB | 3-5GB |
| 推理速度 | 基准 | 相近*（CPU无加速） |
| 精度损失 | 无 | <1%（可接受） |

*CPU 推理速度与 GPU 相比大幅下降，但 INT4 不会进一步减速。

### 量化原理
- INT4 将 32 位浮点数压缩为 4 位整数
- 使用缩放因子（scale）和零点偏移（zero_point）恢复精度
- vLLM 自动处理量化/反量化，应用层透明

---

## 部署步骤

### 1. 准备环境

```bash
# 克隆或进入项目目录
cd /path/to/story_maker

# 创建虚拟环境（如果还未创建）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 确保安装了 vLLM（带 CPU 支持）
pip install vllm
```

### 2. 验证模型路径

确保模型文件位置正确：

```bash
# Linux/macOS:
ls -lh /models/nlg/qwen_2.5_3B/

# Windows:
dir /models/nlg/qwen_2.5_3B\
```

文件应包含：
- `config.json` - 模型配置
- `model-*.safetensors` 或 `pytorch_model.bin` - 权重文件
- `tokenizer.model` 或 `tokenizer.json` - 分词器

### 3. 创建 CPU 配置文件

项目已提供 `.env.vllm.cpu`（包含所有 CPU 优化参数）。

验证其内容：

```bash
cat .env.vllm.cpu
```

关键参数：
```env
MODEL_PATH=/models/nlg/qwen_2.5_3B
VLLM_QUANTIZATION=int4
VLLM_CPU_ONLY=true
VLLM_MAX_PARALLEL_LOADING_WORKERS=1
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.0
```

### 4. 启动 vLLM 服务器

```bash
# 使用 CPU 优化脚本启动
chmod +x scripts/start_vllm_server_cpu.sh
./scripts/start_vllm_server_cpu.sh
```

**首次启动会进行量化初始化**（耗时 2-5 分钟），之后会被缓存，再次启动时更快。

预期输出：
```
=========================================
  vLLM CPU-Only Server Configuration
=========================================
  Model:                /models/nlg/qwen_2.5_3B
  Served as:            qwen-local
  Host:                 127.0.0.1
  Port:                 8000
  Max Model Len:        2048
  Quantization:         int4
  CPU-Only Mode:        true
  Parallel Workers:     1
=========================================

⚠️  NOTE: CPU-only inference is SLOW (5-20s per turn)
    Optimized for memory efficiency, not speed.

Starting vLLM CPU-Only server...
API endpoint: http://127.0.0.1:8000/v1
```

服务器已就绪，当看到以下日志时：
```
Uvicorn running on http://127.0.0.1:8000
```

### 5. 测试推理（另一个终端）

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 运行测试脚本
python scripts/test_openai_api.py
```

输出示例：
```
=== Standard completion ===
Hello! I'm here and ready to help.
⏱️  Inference time: 8.45s

=== Streaming completion ===
Streaming output: 1 2 3
⏱️  Streaming time: 6.32s
```

---

## 性能预期

### 推理延迟（Latency）

基于 AMD Ryzen 7 H255W（8 核）+ 32GB RAM：

| 场景 | 输入长度 | 输出长度 | 推理时间 |
|------|---------|---------|---------|
| 短 | 50 tokens | 50 tokens | 3-5s |
| 中 | 200 tokens | 100 tokens | 8-12s |
| 长 | 1000 tokens | 200 tokens | 20-30s |

**注意**: 实际时间取决于 CPU 型号、系统负载和冷缓存状态。

### 吞吐量（Throughput）

- **单个请求**: ~6-12 tokens/秒（CPU 上限）
- **并发请求**: 不建议超过 2-3 个并发请求（会大幅降速）

### 内存占用

启动时内存使用：

```
总内存: 32GB
├─ 系统 OS: ~2-3GB
├─ Python 进程: ~500MB
├─ 模型（INT4）: ~2.5-3GB
├─ KV 缓存: ~1-2GB（运行时）
└─ 空闲缓冲: ~20-23GB
```

**关键**: 需要至少预留 3-5GB 自由内存用于推理过程。如果其他程序占用大量内存，会导致推理失败或系统卡顿。

---

## 常见问题

### Q1: 推理很慢（20+ 秒）怎么办？

**可能原因和解决方案**:

1. **首次启动**: 第一次运行需要量化初始化，可能耗时 5-10 分钟，属于正常现象。
2. **其他程序占用 CPU**: 关闭浏览器、IDE 等高负载程序。
3. **内存压力**: 检查空闲内存，关闭占用内存的程序。

```bash
# Linux/macOS 检查内存
free -h
top -b -n 1 | head -20

# Windows 检查内存
Get-ComputerInfo | Select-Object CsTotalPhysicalMemory, OsAvailablePhysicalMemory
```

### Q2: vLLM 启动失败，提示"CUDA not found"

**解决方案**:
这是期望的行为，CPU 模式下不需要 CUDA。错误通常不影响启动。

如果服务无法启动，检查：
1. 模型路径是否正确
2. vLLM 版本是否支持 CPU 推理（推荐 >= v0.4.0）

```bash
python -c "import vllm; print(vllm.__version__)"
```

### Q3: 能否使用 FP16（半精度）代替 INT4？

**可能性有限**:
- FP16 模型大小: ~6GB（仍超过可用内存）
- INT4 是 CPU 下的最优选择
- vLLM 在 CPU 上支持 INT4 但对 FP16 的 CPU 支持有限

### Q4: 可以同时处理多个并发请求吗？

**不建议超过 2-3 个**:
- CPU 资源有限，并发增加会导致严重卡顿
- 如需更好的并发性能，必须使用 GPU
- 推荐单次请求处理

### Q5: 如何切换回 GPU 推理？

**步骤**:

1. 确保已安装 CUDA 和 cuDNN
2. 改用原始配置文件：

```bash
# 使用 GPU 版本脚本
./scripts/start_vllm_server.sh

# 或直接修改 .env：
# 重命名或备份 .env.vllm.cpu
mv .env.vllm.cpu .env.vllm.cpu.bak
# 使用 GPU 配置
cp .env.vllm.example .env.vllm
# 修改 MODEL_PATH 指向你的模型
```

---

## 内存占用估算

### 模型加载
```
Qwen 2.5 3B INT4:
  - 权重: ~2.5GB
  - 加载开销: ~200-300MB
  - 总计: ~2.7-2.8GB
```

### 推理时 KV 缓存
```
最大上下文长度 (MAX_MODEL_LEN=2048):
  - KV 缓存大小 ≈ 2 * max_len * hidden_dim * num_layers * dtype_size
  - Qwen 2.5 3B: 约 1-2GB（取决于实际填充）
```

### 总内存预算
```
峰值内存 = 模型 + KV缓存 + 激活 + 系统开销
       ≈ 2.8 + 2.0 + 0.5 + 2.5
       ≈ 7.8-8.5GB
```

**结论**: 32GB 内存足以运行 Qwen 2.5 3B INT4。如果可用内存低于 10GB，可能会导致问题。

---

## 进阶优化（可选）

### CPU 亲和性（Affinity）

在高负载系统上，可以将 vLLM 进程绑定到特定的 CPU 核心：

```bash
# 使用前 4 个核心
taskset -c 0-3 ./scripts/start_vllm_server_cpu.sh
```

### 禁用某些不必要的特性

编辑 `.env.vllm.cpu`，取消注释：

```env
# 禁用详细日志
VLLM_LOG_LEVEL=WARNING

# 如果内存仍然紧张，启用 CPU 卸载
VLLM_CPU_OFFLOAD_GB=4
```

### 使用更小的上下文长度

如果推理仍然缓慢，降低 `MAX_MODEL_LEN`:

```env
# 从 2048 降低到 1024
MAX_MODEL_LEN=1024
```

---

## 故障排除

### 症状: "Out of Memory" (OOM) 错误

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决**:
1. 关闭其他程序，释放内存
2. 降低 `MAX_MODEL_LEN` 到 1024
3. 考虑使用更小的模型（如 1.5B）

### 症状: vLLM 进程消耗 100% CPU，但推理缓慢

这是正常的！CPU 推理总是会用满所有核心。

### 症状: 频繁出现 "connection refused" 错误

```
ConnectionError: Failed to connect to http://127.0.0.1:8000/v1
```

**解决**:
1. 检查 vLLM 服务是否还在运行
2. 检查端口是否被占用：

```bash
# Linux/macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

3. 重启 vLLM 服务

---

## 性能对比

### Qwen 2.5 3B 推理性能对比

| 硬件 | 推理时间（50 in / 50 out） | 吞吐量 |
|------|---------------------------|--------|
| GPU (RTX 4090) | 0.3-0.5s | 100-200 tokens/s |
| GPU (RTX 3080) | 0.8-1.2s | 40-60 tokens/s |
| **CPU (R7 H255W)** | **4-6s** | **8-12 tokens/s** |

---

## 总结

- ✅ **INT4 量化是必需的** - 使 3B 模型适配 32GB 内存
- ✅ **推理速度会显著下降** - 期望 5-20s 延迟（对文本冒险游戏可接受）
- ✅ **首次启动较慢** - 之后会缓存量化结果
- ✅ **并发性能有限** - 不建议超过 2-3 个并发请求
- ✅ **内存占用在可控范围** - 峰值约 8-10GB，远低于 32GB 上限

---

## 相关文件

- 配置文件: `.env.vllm.cpu`
- 启动脚本: `scripts/start_vllm_server_cpu.sh`
- 测试脚本: `scripts/test_openai_api.py`
- GPU 版本: `scripts/start_vllm_server.sh` 和 `.env.vllm`

---

## 参考资源

- [vLLM 官方文档](https://docs.vllm.ai)
- [vLLM CPU 推理指南](https://docs.vllm.ai/en/latest/backends/cpu.html)
- [量化技术介绍](https://huggingface.co/docs/transformers/quantization)
- [Qwen 模型卡](https://huggingface.co/Qwen)
