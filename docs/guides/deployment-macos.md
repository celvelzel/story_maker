# macOS 高可用部署指南（中文）

> **Last Updated**: 2026-04-01

## 1. 脚本说明

生产模式脚本：

1. `scripts/start_project_prod.sh`：生产模式启动（推荐，macOS/Linux）

> **注意**：项目根目录下的 `start_project_prod.sh` 是 `scripts/start_project_prod.sh` 的副本。推荐使用 `scripts/` 目录下的版本。

## 2. 开发模式 vs 生产模式

1. 开发模式：
   - 手动运行：`.venv/bin/python -m streamlit run app.py --server.port=7860`
2. 生产模式：
   - 命令：`./scripts/start_project_prod.sh`
   - 能力：端口检测、安全重启、依赖安装超时、日志落盘

## 3. 首次部署

1. 进入项目根目录。
2. 首次赋权：`chmod +x scripts/start_project_prod.sh`
3. 执行：`./scripts/start_project_prod.sh`
4. 脚本自动完成虚拟环境、依赖安装、端口处理和应用启动。

## 4. llama.cpp 本地推理（可选）

如需使用本地 LLM 推理：

1. 下载 llama.cpp 二进制文件到 `llama.cpp-bin/`
2. 放置 GGUF 模型到 `models/qwen-gguf/qwen3-4b-q4_k_m.gguf`
3. 启动 llama.cpp 服务器：
   ```bash
   ./llama.cpp-bin/llama-server -m models/qwen-gguf/qwen3-4b-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 2048 -b 512 -t 4 --chat-template chatml
   ```
4. 配置 `.env` 指向 `http://127.0.0.1:8081/v1`

详见 [本地模型启动指南](local-model-startup.md)。

## 5. 升级流程

1. 拉取代码后再次执行 `./scripts/start_project_prod.sh`。
2. 关注控制台和 `logs/` 中的启动信息。

## 6. 故障排查

1. 退出码：
   - `1`：通用失败
   - `3`：端口被非本应用进程占用
2. 端口检查：`lsof -i :7860`
3. 日志查看：`tail -n 200 logs/storyweaver_prod_*.log`

## 7. 回滚建议

1. 回到上一个稳定提交版本。
2. 清理或切换虚拟环境后重新启动。
3. 如使用本地 intent checkpoint，同步切换 `models/intent_classifier`。
