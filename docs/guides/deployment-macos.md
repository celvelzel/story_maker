# macOS 高可用部署指南（中文）

## 1. 脚本说明

保留脚本：

1. `start_project.sh`：开发启动（macOS/Linux）

新增脚本：

1. `start_project_prod.sh`：生产模式启动（macOS/Linux）

## 2. 开发模式 vs 生产模式

1. 开发模式：
   - 命令：`./start_project.sh`
2. 生产模式：
   - 命令：`./start_project_prod.sh`
   - 能力：端口检测、安全重启、依赖安装超时、环境变量校验、日志落盘

## 3. 首次部署

1. 进入项目根目录。
2. 首次赋权：`chmod +x start_project_prod.sh`
3. 执行：`./start_project_prod.sh`
4. 脚本自动完成虚拟环境、依赖安装、端口处理和应用启动。

## 4. 升级流程

1. 拉取代码后再次执行 `./start_project_prod.sh`。
2. 关注控制台和 `logs/` 中的启动信息。

## 5. 故障排查

1. 退出码：
   - `1`：通用失败
   - `2`：环境变量缺失
   - `3`：端口被非本应用进程占用
2. 端口检查：`lsof -i :7860`
3. 日志查看：`tail -n 200 logs/storyweaver_prod_*.log`

## 6. 回滚建议

1. 回到上一个稳定提交版本。
2. 清理或切换虚拟环境后重新启动。
3. 如使用本地 intent checkpoint，同步切换 `models/intent_classifier`。
