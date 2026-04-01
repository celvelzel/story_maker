# 本地模型调优与 Bug 修复报告

> **最后更新**：2026-03-31

**日期**：2026-03-27  
**范围**：改善开发者体验、处理推理超时，以及修复本地 llama.cpp 集成的编码问题

---

## 1. 问题识别

在初步集成 `llama.cpp` 进行本地 CPU 推理后，发现了以下问题：
1. **可见性差**：缺乏详细的日志记录，难以追踪请求进度和模型性能
2. **读取超时**：本地 CPU 推理经常超过默认的 60 秒超时，导致 Streamlit 前端频繁出现 `Connection error`（连接错误）失败
3. **编码崩溃**：日志中的 Emoji 字符在某些 Windows 终端上导致 `gbk codec can't encode character`（gbk 编解码器无法编码字符）错误

## 2. 实现细节

### 2.1 增强日志与终端兼容性

修改 `src/utils/api_client.py` 以提供结构化的、终端安全的日志记录：
- **新增日志节点**：
  - `[LLM] RECEIVED request`（收到请求）：显示消息角色和 100 字符的内容预览
  - `[LLM] PROCESSING`（处理中）：显示活动参数（`model`/模型、`temperature`/温度、`max_tokens`/最大 token 数）
  - `[LLM] DONE`（完成）：报告执行时间和 token 使用量（输入/输出）
- **编码修复**：从日志中移除装饰性 Emoji（📥、🔄、✅、⚠️），防止在非 UTF-8 Windows 控制台上崩溃

### 2.2 动态超时配置

为适应基于 CPU 推理较慢的特性，超时设置被移至全局配置：
- **`config.py` 更新**：
  - 新增 `OPENAI_TIMEOUT_CONNECT`（默认：10 秒）—— 连接超时
  - 新增 `OPENAI_TIMEOUT_READ`（默认：60 秒）—— 读取超时
- **本地推理的 `.env` 具体配置**：
  - 将 `OPENAI_TIMEOUT_READ` 增加到 180 秒，用于 llama.cpp 设置，防止长时间生成期间过早断开连接
- **客户端实现**：
  - `src/utils/api_client.py` 现在使用 `httpx.Timeout` 对象，将这些细粒度设置应用于 OpenAI 兼容客户端

### 2.3 文档支持
- 创建 `docs/guides/local-model-startup.md`：本地模型参数和日志解读的快速入门指南
- 创建 `docs/guides/zero-to-hero-deployment.md`：全面的跨平台部署参考

## 3. 验证与结果
- **日志记录**：✅ 已验证终端日志正确显示请求/响应周期
- **编码**：✅ Windows 上不再出现 `gbk` 编解码器错误
- **稳定性**：✅ 本地模型现在可以生成长达 3 分钟的故事片段，而不会被超时中断
- **灵活性**：✅ 已确认 `api_client` 动态遵循 `.env` 超时调整

## 4. 后续步骤
- 监控不同硬件上的 CPU 性能，以优化默认超时值
- 增强 UI 加载状态，明确告知用户何时使用本地模型，以及预期会有更长的响应时间
