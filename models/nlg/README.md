# NLG 模型目录

本目录用于存放微调过的语言模型（LLM），用于本地推理以替换项目的 LLM API 调用。

## 用途

- **本地推理**: 使用本地部署的语言模型替代 OpenAI API 调用
- **数据隐私**: 敏感数据无需上传到第三方服务
- **离线开发**: 无网络环境下仍可进行故事生成和选项生成

## 使用方式

项目通过 `src/nlg/` 模块调用语言模型。要切换到本地模型，需要修改 `config.py` 中的配置或环境变量：

```bash
# .env 配置示例（根据实际模型调整）
NLG_MODEL_TYPE=local
NLG_MODEL_PATH=models/nlg/your-model/
NLG_BASE_URL=http://localhost:8000/v1  # 本地推理服务地址（如 vLLM）
```

## 支持的模型格式

- **vLLM**: 推荐格式，部署命令 `vllm serve models/nlg/your-model/`
- **Hugging Face**: 标准格式，直接加载模型目录
- **GGUF/GGML**: 通过 llama.cpp 量化模型

## 部署示例

### 使用 vLLM 本地部署

```bash
vllm serve models/nlg/your-model/ \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

### API 兼容层

项目 NLG 模块已实现与 OpenAI API 兼容的接口，本地 vLLM 服务可直接替换 `OPENAI_BASE_URL` 配置。
