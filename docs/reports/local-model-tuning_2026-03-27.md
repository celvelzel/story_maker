# 本地模型推理调优与错误修复报告

**日期：** 2026年03月27日

## 1. 任务背景

在上一阶段完成 `llama.cpp` 本地 CPU 推理集成后，系统在实际运行中遇到以下问题：
1. **开发者体验差**：`llama.cpp` 服务端和客户端无明显日志，难以追踪请求进度。
2. **连接超时失败**：本地 CPU 推理较慢，导致 Streamlit 前端默认的 60 秒读取超时频繁触发 `Connection error`。
3. **编码问题**：包含 Emoji 的字符在某些控制台终端引发 `gbk codec can't encode character` 报错。

## 2. 实施改动

### 2.1 日志增强与 Emoji 移除

为了方便演示和 Debug，修改了 `src/utils/api_client.py`，在发送 LLM 请求前后添加了结构化的日志打印：

- **新增日志节点**：
  - `[LLM] RECEIVED request`：展示传入的消息列表，包含 `role` 和前 100 个字符的内容预览。
  - `[LLM] PROCESSING`：显示当前使用的 `model`、`temperature` 和 `max_tokens` 参数。
  - `[LLM] DONE`：展示完成状态、执行耗时以及输入输出的 tokens 数量。
- **兼容性修复**：
  - 去除了最初版本中的 Emoji 符号（📥、🔄、✅、⚠️），以避免 Windows `gbk` 编码终端导致的崩溃问题。

### 2.2 动态超时配置

为了解决本地模型推理较慢引发的超时断连问题，将硬编码的超时设置提取到了全局配置中：

- **修改 `config.py`**：
  - 新增 `OPENAI_TIMEOUT_CONNECT`（默认 10.0秒）。
  - 新增 `OPENAI_TIMEOUT_READ`（默认 60.0秒）。
- **修改 `.env` 配置文件**：
  - 为本地 `llama.cpp` 方案增加了更长的超时参数：
    ```env
    OPENAI_TIMEOUT_CONNECT=30.0
    OPENAI_TIMEOUT_READ=180.0
    ```
- **修改 `src/utils/api_client.py`**：
  - 使用 `httpx.Timeout` 替代了简单的浮点数超时，并将配置项应用于 `OpenAI` 客户端初始化：
    ```python
    import httpx
    self._timeout = httpx.Timeout(self._settings.OPENAI_TIMEOUT_CONNECT, read=self._settings.OPENAI_TIMEOUT_READ)
    ```

### 2.3 增加新文档

- 编写了 `docs/guides/local-model-startup.md`（本地模型启动指南），包含快速启动、参数说明、配置文件和日志演示等信息。
- 编写了 `docs/guides/zero-to-hero-deployment.md`（从零部署指南），作为跨平台的终极部署参考。

## 3. 测试验证

- ✅ **日志打印**：服务器终端能正确打印 `[LLM] RECEIVED request` 等调试信息，方便追踪进度。
- ✅ **编码问题修复**：不再报 `gbk` codec 错误。
- ✅ **超时修复**：本地模型在生成长文本（超过 60s）时，不再被过早打断。
- ✅ **配置读取**：`api_client` 能动态响应 `.env` 文件中的超时设置。

## 4. 下一步计划

1. 根据后续实际的 CPU 性能调整 `.env` 的 `OPENAI_TIMEOUT_READ` 参数。
2. 进一步完善 UI 上的加载状态提示，告知用户当前正在使用本地模型，响应时间可能较长。