# Windows 高可用部署指南（中文）

## 1. 脚本说明

保留脚本：

1. `start_project.bat`：日常开发启动
2. `start_project_simplified.bat`：最小化快速启动

新增脚本：

1. `start_project_prod.bat`：生产模式启动（推荐）

## 2. 开发模式 vs 生产模式

1. 开发模式：
   - 命令：`start_project.bat`
   - 目标：快速迭代
2. 生产模式：
   - 命令：`start_project_prod.bat`
   - 增强：端口检测、安全重启、环境变量校验、日志落盘、明确退出码

## 3. 首次部署

1. 打开 PowerShell，进入项目根目录。
2. 执行：`./start_project_prod.bat`
3. 脚本自动完成：
   - 创建/复用 `.venv`
   - 安装依赖（带网络超时参数）
   - 校验 `.env` 关键变量
   - 检测并处理占用端口进程
   - 启动 Streamlit
4. 日志输出目录：`logs/`

## 4. 升级流程

1. 拉取最新代码。
2. 执行 `start_project_prod.bat` 触发依赖更新。
3. 验证页面可访问与 NLU debug 状态。

## 5. 故障排查

1. 启动失败退出码：
   - `1`：通用失败
   - `2`：环境变量缺失
   - `3`：端口被非本应用进程占用
2. 查看日志：`logs/storyweaver_prod_*.log`
3. 端口排查：`netstat -ano | findstr :7860`

## 6. 回滚建议

1. 保留上一个稳定版本代码包。
2. 回滚后重新执行 `start_project_prod.bat`。
3. 若模型工件有变更，同步回滚 `models/intent_classifier`。
