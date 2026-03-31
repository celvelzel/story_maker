# Windows 高可用部署指南（中文）

> **Last Updated**: 2026-04-01

## 1. 脚本说明

生产模式脚本：

1. `scripts/start_project_prod.bat`：生产模式启动（推荐）

> **注意**：项目根目录下的 `start_project_prod.bat` 是 `scripts/start_project_prod.bat` 的副本。推荐使用 `scripts/` 目录下的版本。

## 2. 开发模式 vs 生产模式

1. 开发模式：
   - 手动运行：`.venv\Scripts\python -m streamlit run app.py --server.port=7860`
2. 生产模式：
   - 命令：`scripts\start_project_prod.bat`
   - 增强：端口检测、安全重启、依赖安装超时、日志落盘、明确退出码

## 3. 首次部署

1. 打开 PowerShell 或 cmd，进入项目根目录。
2. 执行：`scripts\start_project_prod.bat`
3. 脚本自动完成：
   - 创建/复用 `.venv`
   - 安装依赖（带网络超时参数）
   - 检测并处理占用端口进程
   - 启动 Streamlit
4. 日志输出目录：`logs/`

## 4. llama.cpp 本地推理（可选）

如需使用本地 LLM 推理：

1. 下载 llama.cpp 二进制文件到 `llama.cpp-bin\`
2. 放置 GGUF 模型到 `models\qwen-gguf\qwen3-4b-q4_k_m.gguf`
3. 启动 llama.cpp 服务器：
   ```cmd
   scripts\start_llama_server.bat
   ```
4. 配置 `.env` 指向 `http://127.0.0.1:8081/v1`

详见 [本地模型启动指南](local-model-startup.md)。

## 5. 升级流程

1. 拉取最新代码。
2. 执行 `scripts\start_project_prod.bat` 触发依赖更新。
3. 验证页面可访问与 NLU debug 状态。

## 6. 故障排查

1. 启动失败退出码：
   - `1`：通用失败
   - `3`：端口被非本应用进程占用
2. 查看日志：`logs\storyweaver_prod_*.log`
3. 端口排查：`netstat -ano | findstr :7860`

## 7. 回滚建议

1. 保留上一个稳定版本代码包。
2. 回滚后重新执行 `scripts\start_project_prod.bat`。
3. 若模型工件有变更，同步回滚 `models\intent_classifier`。
