# OpenCode Draw.io MCP Server 集成计划

## TL;DR

> **快速总结**: 将 next-ai-draw-io 的 MCP Server **全局安装**到 OpenCode，使你能在所有项目中对话中直接要求 AI 创建和编辑 draw.io 图表。
> 
> **可交付物**:
> - 全局安装 Draw.io MCP Server 包（`npm install -g`）
> - 配置 OpenCode 全局 MCP 设置（跨所有项目）
> - 验证 MCP 连接工作正常
> - 测试从 OpenCode 对话生成图表
> - 创建使用指南文档
> 
> **预期工作量**: Quick | Short | Medium
> **并行执行**: YES - 3 个独立 Wave
> **关键路径**: 全局安装 → 配置 → 启动 → 测试 → 验证

---

## 背景

### 原始需求
用户希望为 OpenCode **全局**集成 next-ai-draw-io 的 MCP Server 功能，使得：
- 可以在**所有 OpenCode 项目**中的对话中要求 AI 创建图表
- 无需在每个项目中重复安装
- 使用官方在线版本（embed.diagrams.net）
- 完整的聊天生成工作流
- MCP Server 作为全局工具对所有项目可用

### 访问信息
- **MCP Server 包**: `@next-ai-drawio/mcp-server@latest`（v0.1.17+）
- **Node 需求**: >= 18
- **默认端口**: 6002（自动递增到 6020 如果端口占用）
- **依赖**: @modelcontextprotocol/sdk, open, linkedom, zod
- **许可证**: Apache-2.0

### 研究发现
MCP Server 功能：
- 完全自包含（内置 HTTP 服务器）
- 支持 Claude Desktop、VS Code、Cursor、Claude Code CLI
- 提供工具: `start_session`、`create_new_diagram`、`edit_diagram`、`get_diagram`、`export_diagram`
- 实时浏览器预览
- 版本历史管理
- 支持自托管 draw.io（通过 DRAWIO_BASE_URL）

---

## 工作目标

### 核心目标
完整集成 Draw.io MCP Server 到 OpenCode，使用户能在对话中创建和编辑图表。

### 具体可交付物
- [ ] 全局安装 @next-ai-drawio/mcp-server@latest（`npm install -g`）
- [ ] OpenCode 全局 MCP 配置文件（`~/.opencode/mcp.json` 或等价配置）
- [ ] MCP 连接验证和调试记录
- [ ] 测试工作流（创建图表、编辑、导出）
- [ ] 使用文档和最佳实践指南
- [ ] 故障排除指南

### 必须包含
- 标准 MCP 配置（npx 启动）
- 环境变量支持（PORT、DRAWIO_BASE_URL）
- 浏览器自动打开机制
- 错误恢复（端口冲突处理）

### 必须排除（Guardrails）
- ❌ 自托管 draw.io 配置（暂不包含）
- ❌ 修改 MCP Server 源代码（使用官方包）
- ❌ 私有 API 密钥集成（MCP Server 本身不需要）
- ❌ 复杂的工作流自动化（仅基础集成）

---

## 验证策略

> **零人工干预** — 所有验证都是代理执行。无例外。

### 测试基础设施
- **已存在**: OpenCode CLI 环境
- **需要配置**: MCP Server 启动脚本
- **测试框架**: 手动 CLI 测试 + 浏览器验证

### QA 策略
每个任务都必须包括代理执行的 QA 场景（参见 TODO 模板）。

**验证方法**:
- **CLI/终端**: 使用 `interactive_bash` 启动 MCP Server
- **浏览器**: 使用 Playwright 打开并验证 draw.io 界面
- **JSON**: 验证 MCP 响应和日志输出

---

## 执行策略

### 并行执行 Wave

```
Wave 1 (立即开始 — 准备和全局安装):
├── Task 1: 验证 Node 环境和 npm 全局安装权限 [quick]
├── Task 2: 全局安装 @next-ai-drawio/mcp-server@latest [quick]
├── Task 3: 创建 OpenCode 全局 MCP 配置文件 (~/.opencode/mcp.json) [quick]
└── Task 4: 创建 MCP 全局启动脚本和检查点 [quick]

Wave 2 (Wave 1 完成后 — 启动和连接):
├── Task 5: 启动全局 MCP Server 进程 [quick]
├── Task 6: 验证 HTTP 服务器就绪 [quick]
├── Task 7: 测试浏览器自动打开机制 [quick]
└── Task 8: 验证 MCP stdio 通信 [quick]

Wave 3 (Wave 2 完成后 — 功能验证和文档):
├── Task 9: 测试创建新图表的完整工作流 [unspecified-high]
├── Task 10: 测试编辑现有图表 [unspecified-high]
├── Task 11: 测试导出和保存图表 [quick]
├── Task 12: 生成使用文档和故障排除指南 [writing]
└── Task 13: 最终集成验证清单 [quick]

Wave FINAL (所有任务完成后 — 最终验证):
├── Task F1: 完整工作流审计 (oracle)
├── Task F2: 配置和依赖检查 (unspecified-high)
├── Task F3: 文档完整性审查 (writing)
└── Task F4: 用户可用性检查 (deep)
-> 向用户展示结果 -> 获得明确的 okay
```

### 依赖矩阵

- **1-4**: — — 5-8
- **5**: 1, 2, 3, 4 — 9-13
- **6**: 5 — 9-13
- **7**: 5, 6 — 9-13
- **8**: 5, 6 — 9-13
- **9**: 5-8 — F1-F4
- **10**: 5-8 — F1-F4
- **11**: 5-8, 9, 10 — F1-F4
- **12**: 5-8, 9-11 — F1-F4
- **13**: 12 — F1-F4
- **F1-F4**: 13 — 用户检查

### 代理分配总结

- **Wave 1**: 4 个任务 → 均为 `quick`
- **Wave 2**: 4 个任务 → 均为 `quick`
- **Wave 3**: 5 个任务 → `unspecified-high` (3), `quick` (1), `writing` (1)
- **Wave FINAL**: 4 个任务 → `oracle` (1), `unspecified-high` (1), `writing` (1), `deep` (1)

---

## TODOs

- [ ] 1. Node 环境和全局安装权限验证

  **要做的事**:
  - 检查 Node.js 版本（需要 >= 18）
  - 验证 npm 可用且能访问 npmjs.com
  - 检查用户有权限进行全局安装（`npm install -g`）
  - 检查全局安装路径可写入
  - 测试 npx 命令工作正常

  **必须不做**:
  - ❌ 安装全局 Node 版本（使用现有环境）
  - ❌ 修改系统 PATH
  - ❌ 使用 sudo（应避免需要权限提升）

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 简单的环境检查，无依赖
  - **技能**: [`bash`]
    - `bash`: 运行 node、npm、npx 版本检查

  **并行化**:
  - **可并行运行**: YES
  - **并行组**: Wave 1
  - **阻塞**: Task 2, 3
  - **被阻塞**: 无

  **参考**（关键）:

  > 执行器没有来自访问的上下文。参考是他们的唯一指南。
  > 每个参考必须回答："我应该看什么以及为什么?"

  **模式参考** (现有代码遵循):
  - OpenCode 项目中的其他全局配置

  **API/类型参考** (实现的契约):
  - Node.js 官方文档: v18+ LTS 支持
  - npm 全局安装：https://docs.npmjs.com/cli/v10/commands/npm-install — `-g` 标志

  **测试参考** (遵循的测试模式):
  - 无 — 这是环境检查

  **外部参考** (库和框架):
  - 官方文档: https://nodejs.org/en/download/ — Node 18+ 安装指南
  - npm 文档: https://docs.npmjs.com/ — 全局安装权限

  **每个参考为什么重要** (解释相关性):
  - MCP Server 需要 Node 18+ 才能运行
  - npx 用于启动 MCP Server 包
  - 全局安装意味着跨所有项目共享
  - 网络访问需要安装 @next-ai-drawio/mcp-server 包

  **验收标准**:
  - [ ] `node --version` 输出 v18.0.0+
  - [ ] `npm --version` 输出有效版本号
  - [ ] `npx --version` 命令可用
  - [ ] `npm config get prefix` 返回可写入路径
  - [ ] `npm ping` 返回成功（网络连接）

  **QA 场景（必须 — 没有这些的任务会被拒绝）**:

  > 这不是可选的。没有 QA 场景的任务将被拒绝。
  >
  > 编写场景测试来验证你构建的东西的实际行为。
  > 最少: 1 个成功路径 + 每个任务 1 个失败/边界情况。
  > 每个场景 = 精确工具 + 精确步骤 + 精确断言 + 证据路径。
  >
  > **执行代理必须在实现后运行这些场景。**
  > **编排器将在标记任务完成前验证证据文件存在。**

  ```
  场景: Node 版本检查通过
    工具: Bash
    前置条件: 系统安装了 Node.js
    步骤:
      1. 运行 `node --version`
      2. 解析输出版本号
      3. 比较版本号 >= v18.0.0
    预期结果: 版本号 >= v18.0.0，exit code 0
    故障指示器: 版本 < v18 或命令不存在
    证据: .sisyphus/evidence/task-1-node-version.txt

  场景: 全局安装权限检查
    工具: Bash
    前置条件: npm 已安装
    步骤:
      1. 运行 `npm config get prefix` 获取全局安装路径
      2. 检查路径是否存在且可写入
      3. 测试 `npm list -g --depth=0` 可读取全局包
    预期结果: 全局路径可访问，可列出全局包
    故障指示器: 路径不存在或无权限
    证据: .sisyphus/evidence/task-1-npm-global.txt

  场景: npm 包访问失败处理
    工具: Bash
    前置条件: npm 注册表不可访问（模拟离线）
    步骤:
      1. 运行 `npm ping` 不带网络
      2. 断言失败的错误信息清晰
    预期结果: 清晰的错误消息，建议检查网络
    证据: .sisyphus/evidence/task-1-npm-offline.txt
  ```

  > **特异性要求 — 每个场景必须使用**:
  > - **选择器**: 特定的 CSS 选择器（`.login-button`，不是"登录按钮"）
  > - **数据**: 具体测试数据（`"test@example.com"`，不是 `"[email]"`）
  > - **断言**: 精确值（`文本包含"欢迎回来"`，不是"验证它工作"）
  > - **时间**: 相关的等待条件（`超时: 10s`）
  > - **负面**: 每个任务至少一个失败/错误场景
  >
  > **反模式（如果看起来像这样，你的场景无效）**:
  > - ❌ "验证它正确工作" — 如何？什么是"正确"？
  > - ❌ "检查 API 返回数据" — 什么数据？什么字段？什么值？
  > - ❌ "测试组件呈现" — 在哪里？什么选择器？什么内容？
  > - ❌ 任何没有证据路径的场景

  **证据捕获**:
  - [ ] 每个证据文件命名: task-{N}-{scenario-slug}.{ext}
  - [ ] 用于 UI 的屏幕截图，用于 CLI 的终端输出，用于 API 的响应体

  **提交**: NO (单个提交分组 Wave 1)

---

- [ ] 2. 全局安装 MCP Server 包

  **要做的事**:
  - 运行 `npm install -g @next-ai-drawio/mcp-server@latest`
  - 验证全局安装成功
  - 检查 `next-ai-drawio-mcp` 命令可从任何目录执行
  - 验证版本 >= 0.1.17
  - 测试所有依赖已全局解决

  **必须不做**:
  - ❌ 本地安装（不用 `-g`）— 这是全局集成
  - ❌ 固定版本号（保持 @latest）— 易于更新
  - ❌ 在特定项目目录中安装

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: npm 全局包安装验证
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: YES
  - **并行组**: Wave 1
  - **阻塞**: Task 5, 6
  - **被阻塞**: Task 1

  **参考**:

  **API/类型参考**:
  - package.json: https://raw.githubusercontent.com/DayuanJiang/next-ai-draw-io/main/packages/mcp-server/package.json — 全局 bin 入口点
  - npm install -g: https://docs.npmjs.com/cli/v10/commands/npm-install

  **外部参考**:
  - npm docs: https://docs.npmjs.com/cli/v10/commands/npm-install — 全局安装指南

  **验收标准**:
  - [ ] 全局命令可用: `which next-ai-drawio-mcp` 或 `where next-ai-drawio-mcp`（Windows）
  - [ ] `npm list -g @next-ai-drawio/mcp-server` 显示已安装
  - [ ] 版本检查显示 >= 0.1.17
  - [ ] 所有依赖（@modelcontextprotocol/sdk, open, linkedom, zod）全局可用

  **QA 场景**:

  ```
  场景: 全局包安装成功
    工具: Bash
    前置条件: npm 可用，网络连接正常，有全局安装权限
    步骤:
      1. 运行 `npm install -g @next-ai-drawio/mcp-server@latest`
      2. 等待安装完成
      3. 运行 `npm list -g @next-ai-drawio/mcp-server`
      4. 检查版本号
    预期结果: 包安装到全局目录，版本 >= 0.1.17
    故障指示器: "找不到包" 或 "权限被拒绝"
    证据: .sisyphus/evidence/task-2-npm-install-global.txt

  场景: 全局命令可执行
    工具: Bash
    前置条件: 包已全局安装
    步骤:
      1. 运行 `next-ai-drawio-mcp --version` 或 `npx @next-ai-drawio/mcp-server --version`
      2. 检查输出包含版本号
    预期结果: 命令可执行，返回版本号
    故障指示器: "命令不找" 或 "权限被拒绝"
    证据: .sisyphus/evidence/task-2-mcp-command.txt

  场景: 从任意目录可调用
    工具: Bash
    前置条件: 命令已安装
    步骤:
      1. 从临时目录（如 `/tmp` 或用户主目录）运行命令
      2. 验证命令仍然可用
    预期结果: 从任何目录可执行
    证据: .sisyphus/evidence/task-2-cmd-anywhere.txt
  ```

  **提交**: NO (Wave 1 end)

---

- [ ] 3. 创建 OpenCode 全局 MCP 配置文件

  **要做的事**:
  - 创建全局 OpenCode MCP 配置（`~/.opencode/mcp.json` 或等价全局配置路径）
  - 配置 Draw.io MCP Server 全局入口（使用全局命令或 npx）
  - 添加环境变量支持（PORT、DRAWIO_BASE_URL）
  - 验证 JSON 格式正确
  - 确保配置对所有项目生效

  **必须不做**:
  - ❌ 硬编码端口号（使用默认 6002 让系统自动递增）
  - ❌ 修改 OpenCode 核心配置（仅添加新 MCP）
  - ❌ 包含 API 密钥（不需要用于 MCP）
  - ❌ 项目本地配置（这是全局设置）

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 简单的 JSON 配置创建
  - **技能**: [`bash`]
    - `bash`: 创建全局目录和文件

  **并行化**:
  - **可并行运行**: YES (与 Task 1, 2 无依赖)
  - **并行组**: Wave 1
  - **阻塞**: Task 4, 5
  - **被阻塞**: Task 1, 2 (需要 Node 检查和全局安装)

  **参考**:

  **模式参考**:
  - MCP Server README: https://github.com/DayuanJiang/next-ai-draw-io/blob/main/packages/mcp-server/README.md — 官方配置示例
  - Claude Desktop 全局配置位置：`~/Library/Application Support/Claude/claude_desktop_config.json`

  **API/类型参考**:
  - OpenCode MCP 全局配置格式（标准 JSON Schema）
  - 全局配置路径惯例

  **外部参考**:
  - 官方文档: https://modelcontextprotocol.io/docs — MCP 配置标准

  **验收标准**:
  - [ ] 全局配置文件创建: `~/.opencode/mcp.json`
  - [ ] JSON 格式正确（无语法错误）
  - [ ] 包含必须字段: `mcpServers.drawio.command` 指向全局命令
  - [ ] 支持环境变量配置
  - [ ] 配置对所有项目可见

  **QA 场景**:

  ```
  场景: 全局配置文件有效且可解析
    工具: Bash
    前置条件: ~/.opencode/mcp.json 已创建
    步骤:
      1. 运行 `cat ~/.opencode/mcp.json | jq .` 验证 JSON 格式
      2. 检查存在 mcpServers.drawio 字段
      3. 验证 command 使用全局命令或 npx
      4. 验证 args 包含 "@next-ai-drawio/mcp-server@latest"
    预期结果: JSON 有效，所有字段存在，exit code 0
    故障指示器: JSON 语法错误或字段缺失
    证据: .sisyphus/evidence/task-3-global-config.json

  场景: 全局配置在多个项目中可用
    工具: Bash
    前置条件: 全局配置已创建
    步骤:
      1. 从不同项目目录读取 ~/.opencode/mcp.json
      2. 验证配置相同且可访问
      3. 检查配置中的 drawio 入口点
    预期结果: 从任何项目目录可读取配置
    证据: .sisyphus/evidence/task-3-global-available.txt

  场景: 配置支持环境变量覆盖
    工具: Bash
    前置条件: 配置文件存在
    步骤:
      1. 检查配置中 env 部分的格式
      2. 验证可以设置 PORT=6003
      3. 验证可以设置 DRAWIO_BASE_URL
    预期结果: 环境变量格式正确，可覆盖
    证据: .sisyphus/evidence/task-3-global-env.txt
  ```

  **提交**: NO (Wave 1 end)

  **要做的事**:
  - 创建启动脚本 (shell/batch 脚本)
  - 添加健康检查（端口可用性、响应性）
  - 实现日志记录到文件
  - 添加优雅关闭处理
  - 创建快速参考指南

  **必须不做**:
  - ❌ 复杂的进程管理（仅基础启动）
  - ❌ 依赖 PM2 或其他守护进程工具

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 简单脚本创建
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: YES
  - **并行组**: Wave 1
  - **阻塞**: Task 5
  - **被阻塞**: Task 1-3

  **参考**:

  **模式参考**:
  - MCP Server 端口处理: README 提到自动递增到 6020

  **验收标准**:
  - [ ] 脚本创建: `scripts/start-drawio-mcp.sh` (Linux/Mac) 或 `.bat` (Windows)
  - [ ] 脚本可执行且有清晰的使用说明
  - [ ] 包含健康检查（至少 3 次重试）
  - [ ] 日志输出到 `.sisyphus/logs/drawio-mcp.log`

  **QA 场景**:

  ```
  场景: 启动脚本可执行
    工具: Bash
    前置条件: 脚本文件已创建
    步骤:
      1. 检查脚本权限 (chmod +x)
      2. 运行 `bash scripts/start-drawio-mcp.sh` 或对应脚本
      3. 验证初始化消息输出
    预期结果: 脚本启动无错误，输出初始化日志
    证据: .sisyphus/evidence/task-4-startup-script.txt

  场景: 健康检查工作
    工具: Bash
    前置条件: 脚本实现了健康检查
    步骤:
      1. 查看脚本中的健康检查逻辑
      2. 验证包含端口可用性检查
      3. 验证包含最多 3 次重试
    预期结果: 健康检查逻辑完整
    证据: .sisyphus/evidence/task-4-health-check.txt
  ```

  **提交**: NO (Wave 1 end)

---

- [ ] 5. 启动 MCP Server 进程

  **要做的事**:
  - 执行启动脚本或 `npx @next-ai-drawio/mcp-server@latest`
  - 验证进程启动成功
  - 记录 PID 和端口信息
  - 验证 stderr/stdout 无错误
  - 等待"就绪"消息

  **必须不做**:
  - ❌ 阻塞执行（在后台运行）
  - ❌ 硬编码端口 (让系统选择)

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 进程启动和初始化
  - **技能**: [`bash`, `interactive_bash`]
    - `bash`: 启动进程和日志收集
    - `interactive_bash`: tmux 会话管理

  **并行化**:
  - **可并行运行**: NO
  - **并行组**: Wave 2 (顺序)
  - **阻塞**: Task 6, 7, 8
  - **被阻塞**: Task 1-4

  **参考**:

  **模式参考**:
  - MCP Server 启动: `npx @next-ai-drawio/mcp-server@latest`
  - 日志输出格式（来自官方包）

  **验收标准**:
  - [ ] 进程启动且运行中（ps 显示）
  - [ ] 没有错误日志（stderr 为空或仅警告）
  - [ ] 输出包含"Server ready"或类似消息
  - [ ] 进程 PID 记录到 `.sisyphus/logs/mcp.pid`

  **QA 场景**:

  ```
  场景: MCP Server 成功启动
    工具: interactive_bash (tmux)
    前置条件: 配置和启动脚本就绪
    步骤:
      1. 启动新 tmux 会话: `tmux new-session -d -s drawio-mcp`
      2. 发送启动命令: `npx @next-ai-drawio/mcp-server@latest`
      3. 等待 2 秒并检查输出
      4. 验证进程运行中: `ps aux | grep mcp-server`
    预期结果: 进程运行中，无错误日志
    故障指示器: "Error"、"failed"、"crashed" 在输出中
    证据: .sisyphus/evidence/task-5-mcp-startup.txt

  场景: 端口自动选择工作
    工具: Bash
    前置条件: MCP Server 运行中
    步骤:
      1. 从日志提取实际端口
      2. 检查端口范围 6002-6020
      3. 运行 `lsof -i :{port}` 验证端口占用
    预期结果: 端口在有效范围内且被 MCP 进程占用
    证据: .sisyphus/evidence/task-5-port-selection.txt
  ```

  **提交**: NO (Wave 2 end)

---

- [ ] 6. 验证 HTTP 服务器就绪

  **要做的事**:
  - 对嵌入的 HTTP 服务器进行健康检查
  - 测试根路由响应
  - 验证响应头和 CORS
  - 检查静态资源可用（HTML、CSS、JS）
  - 验证 MCP session 路由工作

  **必须不做**:
  - ❌ 打开浏览器（仅 CLI 验证）

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: HTTP 健康检查
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: NO (需要 Task 5)
  - **并行组**: Wave 2 (顺序)
  - **阻塞**: Task 7, 8
  - **被阻塞**: Task 5

  **参考**:

  **API/类型参考**:
  - HTTP 响应格式 (标准 HTTP/1.1)
  - Content-Type: text/html

  **外部参考**:
  - curl 文档: https://curl.se/ — HTTP 测试工具

  **验收标准**:
  - [ ] `curl http://localhost:6002` 返回 200 状态码
  - [ ] 响应包含 HTML 内容
  - [ ] 响应头包含 Content-Type: text/html
  - [ ] 没有 5xx 错误

  **QA 场景**:

  ```
  场景: HTTP 服务器响应正常
    工具: Bash
    前置条件: MCP Server 运行中
    步骤:
      1. 运行 `curl -v http://localhost:6002`
      2. 检查状态码是否为 200
      3. 检查响应包含 HTML
    预期结果: 状态 200，HTML 响应体
    故障指示器: 状态 != 200 或空响应
    证据: .sisyphus/evidence/task-6-http-health.txt

  场景: 健康检查端点可用
    工具: Bash
    前置条件: HTTP 服务器就绪
    步骤:
      1. 运行 `curl -I http://localhost:6002/health` 或检查日志
      2. 验证服务器在线
    预期结果: 服务器响应（200 或其他 2xx）
    证据: .sisyphus/evidence/task-6-health-endpoint.txt
  ```

  **提交**: NO (Wave 2 end)

---

- [ ] 7. 测试浏览器自动打开机制

  **要做的事**:
  - 验证 `start_session` 工具打开浏览器
  - 确认浏览器打开到正确的 URL
  - 验证 MCP session ID 在 URL 中（?mcp=xxx）
  - 检查 draw.io UI 加载
  - 验证浏览器关闭时优雅处理

  **必须不做**:
  - ❌ 实际交互（仅打开）
  - ❌ 修改浏览器配置

  **推荐代理配置**:
  - **类别**: `visual-engineering`
    - 理由: 浏览器 UI 测试
  - **技能**: [`playwright`]

  **并行化**:
  - **可并行运行**: NO (需要 Task 5, 6)
  - **并行组**: Wave 2 (顺序)
  - **阻塞**: Task 9, 10
  - **被阻塞**: Task 5, 6

  **参考**:

  **模式参考**:
  - Playwright 浏览器控制: 标准 Playwright API

  **外部参考**:
  - Playwright 文档: https://playwright.dev/ — 浏览器自动化

  **验收标准**:
  - [ ] 浏览器打开成功（通过 `open` 包或自动启动）
  - [ ] URL 包含 localhost:6002 和 ?mcp= 参数
  - [ ] draw.io 界面在浏览器中可见
  - [ ] 没有加载错误（Console 中无红色错误）

  **QA 场景**:

  ```
  场景: 浏览器自动打开到正确 URL
    工具: Playwright
    前置条件: MCP Server 运行，HTTP 服务器就绪
    步骤:
      1. 使用 Playwright 打开 http://localhost:6002
      2. 等待页面加载（timeout 10s）
      3. 检查 URL 包含 ?mcp= 参数
      4. 截图验证 draw.io UI 存在
    预期结果: 页面加载成功，URL 正确，UI 可见
    故障指示器: 页面空白、404、加载失败
    证据: .sisyphus/evidence/task-7-browser-open.png

  场景: draw.io 编辑器界面加载
    工具: Playwright
    前置条件: 浏览器打开
    步骤:
      1. 等待 draw.io 编辑器加载
      2. 查找编辑器画布元素
      3. 查找工具栏和菜单
    预期结果: 所有编辑器元素加载完成
    证据: .sisyphus/evidence/task-7-drawio-ui.png
  ```

  **提交**: NO (Wave 2 end)

---

- [ ] 8. 验证 MCP stdio 通信

  **要做的事**:
  - 测试 MCP 工具调用通信
  - 验证 `get_diagram` 工具返回正确的 XML
  - 验证错误处理和响应格式
  - 测试 JSON-RPC 2.0 协议兼容性
  - 检查 stdio 缓冲和流处理

  **必须不做**:
  - ❌ 修改 MCP 协议实现
  - ❌ 测试 AI 推理（仅协议）

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 协议验证
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: NO
  - **并行组**: Wave 2 (顺序)
  - **阻塞**: Task 9, 10
  - **被阻塞**: Task 5, 6

  **参考**:

  **API/类型参考**:
  - MCP 工具定义: `get_diagram`, `create_new_diagram`, `edit_diagram`, `export_diagram`, `start_session`
  - JSON-RPC 2.0 格式

  **外部参考**:
  - MCP 规范: https://modelcontextprotocol.io/docs — 协议文档

  **验收标准**:
  - [ ] MCP 工具调用通过 stdio 传输
  - [ ] 响应格式正确（JSON）
  - [ ] `get_diagram` 返回有效的 draw.io XML
  - [ ] 没有协议错误

  **QA 场景**:

  ```
  场景: MCP 工具调用返回有效响应
    工具: Bash
    前置条件: MCP Server 和 HTTP 服务器运行中
    步骤:
      1. 发送 MCP 工具调用（模拟 get_diagram）
      2. 捕获 JSON 响应
      3. 验证 JSON 格式正确
    预期结果: 有效的 JSON-RPC 2.0 响应
    故障指示器: 无效 JSON 或错误代码
    证据: .sisyphus/evidence/task-8-mcp-protocol.txt

  场景: XML 响应有效
    工具: Bash
    前置条件: get_diagram 工具调用成功
    步骤:
      1. 解析响应中的 XML
      2. 验证 XML 格式正确（draw.io 格式）
      3. 检查必须的元素存在
    预期结果: 有效的 draw.io XML
    证据: .sisyphus/evidence/task-8-xml-response.xml
  ```

  **提交**: NO (Wave 2 end)

---

- [ ] 9. 测试创建新图表的完整工作流

  **要做的事**:
  - 启动 MCP session
  - 在浏览器中创建新图表
  - 验证图表在编辑器中显示
  - 测试添加形状、连接器等基本元素
  - 验证实时更新
  - 导出图表并验证文件

  **必须不做**:
  - ❌ 测试高级 AI 推理（仅创建工作流）
  - ❌ 图表美观测试

  **推荐代理配置**:
  - **类别**: `unspecified-high`
    - 理由: 完整工作流测试
  - **技能**: [`playwright`]

  **并行化**:
  - **可并行运行**: NO (顺序依赖)
  - **并行组**: Wave 3
  - **阻塞**: Task 12, 13
  - **被阻塞**: Task 5-8

  **参考**:

  **模式参考**:
  - draw.io 图表 XML 格式
  - MCP 工具: `create_new_diagram`, `export_diagram`

  **验收标准**:
  - [ ] 新图表成功创建
  - [ ] 图表在浏览器编辑器中显示
  - [ ] 可添加基本元素（矩形、圆形等）
  - [ ] 导出产生有效的 .drawio 文件

  **QA 场景**:

  ```
  场景: 创建新图表并在编辑器显示
    工具: Playwright
    前置条件: MCP session 启动，浏览器打开
    步骤:
      1. 调用 create_new_diagram 工具
      2. 等待浏览器更新（2 秒）
      3. 验证编辑器中出现空白画布
      4. 截图验证
    预期结果: 空白图表在编辑器中显示
    故障指示器: 页面空白、错误或加载失败
    证据: .sisyphus/evidence/task-9-new-diagram.png

  场景: 导出图表成功
    工具: Bash
    前置条件: 图表已创建
    步骤:
      1. 调用 export_diagram 工具
      2. 验证文件保存成功
      3. 检查文件大小 > 0
      4. 验证 XML 格式
    预期结果: 有效的 .drawio 文件导出
    证据: .sisyphus/evidence/task-9-export.drawio
  ```

  **提交**: NO (Wave 3 end)

---

- [ ] 10. 测试编辑现有图表

  **要做的事**:
  - 加载现有图表（从文件或示例）
  - 测试 `edit_diagram` 工具进行修改
  - 验证版本历史跟踪
  - 测试恢复到先前版本
  - 验证编辑同步到浏览器

  **必须不做**:
  - ❌ AI 推理驱动的编辑
  - ❌ 复杂的多人编辑

  **推荐代理配置**:
  - **类别**: `unspecified-high`
    - 理由: 编辑和版本控制工作流
  - **技能**: [`playwright`, `bash`]

  **并行化**:
  - **可并行运行**: YES (与 Task 11 独立)
  - **并行组**: Wave 3
  - **阻塞**: Task 12, 13
  - **被阻塞**: Task 5-8

  **参考**:

  **模式参考**:
  - MCP 工具: `edit_diagram`, `get_diagram`
  - 版本历史管理

  **验收标准**:
  - [ ] 现有图表加载成功
  - [ ] 编辑工具可修改图表
  - [ ] 版本历史保留先前状态
  - [ ] 恢复功能工作正常

  **QA 场景**:

  ```
  场景: 编辑现有图表
    工具: Playwright
    前置条件: 图表已创建
    步骤:
      1. 加载现有图表
      2. 修改元素（如更改标签）
      3. 验证编辑器显示更改
      4. 保存修改
    预期结果: 修改成功保存
    证据: .sisyphus/evidence/task-10-edit-diagram.png

  场景: 版本历史恢复工作
    工具: Playwright
    前置条件: 图表经历多个编辑
    步骤:
      1. 打开版本历史（时钟按钮）
      2. 选择先前版本
      3. 验证图表恢复到该状态
    预期结果: 图表成功恢复
    证据: .sisyphus/evidence/task-10-version-restore.png
  ```

  **提交**: NO (Wave 3 end)

---

- [ ] 11. 测试导出和保存图表

  **要做的事**:
  - 测试多种导出格式（.drawio XML、可能的 PNG/SVG）
  - 验证导出文件质量
  - 测试文件保存到指定位置
  - 验证文件元数据（大小、创建时间等）
  - 测试重复导出覆盖

  **必须不做**:
  - ❌ 图像导出（仅 XML 格式）
  - ❌ 云存储集成

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 文件 I/O 和格式验证
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: YES (与 Task 9, 10 独立)
  - **并行组**: Wave 3
  - **阻塞**: Task 12, 13
  - **被阻塞**: Task 5-8, 9-10

  **参考**:

  **API/类型参考**:
  - draw.io XML 格式规范
  - MCP 工具: `export_diagram`

  **验收标准**:
  - [ ] 导出成功产生 .drawio 文件
  - [ ] 文件大小 > 0 字节
  - [ ] 文件可在 draw.io 中打开
  - [ ] 文件包含所有图表数据

  **QA 场景**:

  ```
  场景: 导出文件有效并可重新打开
    工具: Bash
    前置条件: 图表已导出到文件
    步骤:
      1. 验证文件存在
      2. 检查文件大小
      3. 验证 XML 格式正确
      4. 解析 XML 根元素
    预期结果: 有效的 .drawio XML 文件
    故障指示器: 文件不存在、为空或格式无效
    证据: .sisyphus/evidence/task-11-export-file.txt

  场景: 重复导出覆盖旧文件
    工具: Bash
    前置条件: 图表已导出一次
    步骤:
      1. 再次导出到同一文件
      2. 验证新文件替换旧文件
      3. 对比时间戳
    预期结果: 新文件覆盖旧文件
    证据: .sisyphus/evidence/task-11-export-overwrite.txt
  ```

  **提交**: NO (Wave 3 end)

---

- [ ] 12. 生成使用文档和故障排除指南

  **要做的事**:
  - 创建 `DRAW_IO_MCP_USAGE.md` 用户指南
  - 文档化如何从 OpenCode 对话使用工具
  - 创建示例提示和预期输出
  - 编写常见问题和故障排除部分
  - 添加配置选项和高级用法
  - 创建故障排除检查表

  **必须不做**:
  - ❌ 复制官方文档（仅链接）
  - ❌ 不必要的冗长

  **推荐代理配置**:
  - **类别**: `writing`
    - 理由: 文档创建和编辑
  - **技能**: [`write`, `edit`]

  **并行化**:
  - **可并行运行**: YES (与其他独立)
  - **并行组**: Wave 3
  - **阻塞**: Task 13
  - **被阻塞**: Task 5-11

  **参考**:

  **模式参考**:
  - MCP Server README: https://github.com/DayuanJiang/next-ai-draw-io/blob/main/packages/mcp-server/README.md
  - OpenCode 文档样式

  **验收标准**:
  - [ ] 文档文件创建: `.sisyphus/docs/DRAW_IO_MCP_USAGE.md`
  - [ ] 包含至少 5 个用户示例
  - [ ] 故障排除部分至少 3 个常见问题
  - [ ] 包含配置参考
  - [ ] Markdown 格式正确

  **QA 场景**:

  ```
  场景: 文档完整且格式正确
    工具: Bash
    前置条件: 文档已创建
    步骤:
      1. 检查文件存在: .sisyphus/docs/DRAW_IO_MCP_USAGE.md
      2. 验证 Markdown 格式（标题、代码块等）
      3. 运行 markdown linter（如有）
    预期结果: 有效的 Markdown 文件，无格式错误
    故障指示器: 文件不存在、格式破损
    证据: .sisyphus/evidence/task-12-doc-check.txt

  场景: 文档包含可执行示例
    工具: Bash
    前置条件: 文档已创建
    步骤:
      1. 提取文档中的代码示例
      2. 验证语法正确
      3. 检查示例具体可用
    预期结果: 所有示例都是具体、可执行的
    证据: .sisyphus/evidence/task-12-examples.txt
  ```

  **提交**: NO (Wave 3 end)

---

- [ ] 13. 最终集成验证清单

  **要做的事**:
  - 验证所有组件完整性
  - 检查配置端到端工作
  - 验证日志和错误处理
  - 检查文件和目录结构
  - 准备最终部署清单

  **必须不做**:
  - ❌ 修改已验证的组件
  - ❌ 超出范围的功能测试

  **推荐代理配置**:
  - **类别**: `quick`
    - 理由: 检查表验证
  - **技能**: [`bash`]

  **并行化**:
  - **可并行运行**: NO
  - **并行组**: Wave 3 (最后)
  - **阻塞**: F1-F4
  - **被阻塞**: Task 12

  **验收标准**:
  - [ ] 所有 Wave 1-3 任务完成
  - [ ] 无遗留错误或警告
  - [ ] 所有文档和配置就位
  - [ ] 最终部署检查表 100% 完成

  **QA 场景**:

  ```
  场景: 所有配置文件到位
    工具: Bash
    前置条件: 所有任务完成
    步骤:
      1. 验证文件存在:
         - .opencode/mcp.json
         - 启动脚本
         - 文档
      2. 检查文件权限和可读性
    预期结果: 所有文件存在且可访问
    证据: .sisyphus/evidence/task-13-file-manifest.txt

  场景: 集成验证检查表
    工具: Bash
    前置条件: 所有组件完成
    步骤:
      1. 运行最终检查列表
      2. 验证 100% 检查通过
    预期结果: 完整集成就绪
    证据: .sisyphus/evidence/task-13-final-checklist.txt
  ```

  **提交**: YES (完整部署包)
  - 提交消息: `feat(mcp): integrate draw.io MCP server into OpenCode with full documentation`
  - 文件: `.opencode/mcp.json`, `scripts/start-drawio-mcp.*`, `.sisyphus/docs/DRAW_IO_MCP_USAGE.md`, 等
  - 预提交: `npm run lint` (如果适用)

---

## 最终验证 Wave（所有实现任务完成后 — 4 个并行审查，然后用户 okay）

> 4 个审查代理并行运行。所有必须批准。向用户展示结果并获得明确的 okay 后才能完成。
>
> **勿自动进行验证后流程。等待用户的明确批准后才能标记工作完成。**
> **勿在标记 F1-F4 为已检查前获得用户的 okay。** 拒绝或用户反馈 -> 修复 -> 重新运行 -> 展示 -> 等待 okay。

- [ ] F1. **集成部署审计** — `oracle`
  逐一阅读计划。对于每个"必须包含"：验证实现存在（读取文件、检查配置、运行命令）。对于每个"必须排除"：在代码库中搜索禁止模式 — 如果找到则拒绝（file:line）。检查证据文件存在在 .sisyphus/evidence/。比较可交付物与计划。
  输出: `Must Have [N/N] | Must NOT Have [N/N] | 任务 [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **MCP 配置和连接检查** — `unspecified-high`
  验证 .opencode/mcp.json 格式正确。测试启动脚本可执行。运行 `npx @next-ai-drawio/mcp-server@latest --version`。检查日志无错误。验证 HTTP 服务器端口绑定。
  输出: `配置 [PASS/FAIL] | 脚本 [PASS/FAIL] | 包版本 [PASS/FAIL] | 连接 [PASS/FAIL] | VERDICT`

- [ ] F3. **端到端工作流测试** — `unspecified-high` (+ `playwright` 技能)
  从头开始执行：启动 MCP Server → 打开浏览器 → 创建新图表 → 编辑 → 导出。每步验证无错误。保存到 `.sisyphus/evidence/final-qa/`。
  输出: `工作流 [PASS/FAIL] | 步骤 [N/N] | 导出 [PASS/FAIL] | VERDICT`

- [ ] F4. **文档和用户可用性检查** — `deep`
  读取生成的文档（DRAW_IO_MCP_USAGE.md）。验证：示例准确、说明清晰、故障排除有用。检查与实际行为的一致性。从文档推荐中实际尝试示例。
  输出: `文档 [清晰/不清晰] | 示例 [可用/不可用] | 故障排除 [有用/无用] | VERDICT`

---

## 提交策略

### Wave 1 结束 — 配置和准备

- **提交消息**: `chore(mcp): setup OpenCode draw.io MCP configuration and startup scripts`
- **文件**: `.opencode/mcp.json`, `scripts/start-drawio-mcp.sh/bat`, `.sisyphus/logs/` (empty)
- **预提交**: JSON 格式验证

### Wave 2 结束 — 启动和验证

- **提交消息**: `feat(mcp): verify MCP server startup and HTTP connectivity`
- **文件**: 启动脚本更新（如有）, `.sisyphus/logs/drawio-mcp.log`
- **预提交**: 日志检查（无错误）

### Wave 3 结束 — 功能和文档

- **提交消息**: `docs(mcp): complete draw.io MCP integration with usage guide and QA`
- **文件**: `.sisyphus/docs/DRAW_IO_MCP_USAGE.md`, `.sisyphus/evidence/` (所有 QA 证据)
- **预提交**: Markdown lint

### 最终验证完成

- **提交消息**: `feat(mcp): draw.io MCP integration complete and verified`
- **文件**: 合并所有更改
- **标记**: 版本标签（可选）

---

## 成功标准

### 验证命令

```bash
# 检查配置
cat .opencode/mcp.json | jq .

# 验证包可用
npx @next-ai-drawio/mcp-server@latest --version

# 检查 HTTP 服务器
curl http://localhost:6002

# 验证导出
ls -la *.drawio
```

### 最终检查清单

- [x] MCP 配置文件有效
- [x] 启动脚本可执行
- [x] HTTP 服务器响应
- [x] MCP 工具调用工作
- [x] 浏览器打开正常
- [x] 图表创建/编辑/导出功能完整
- [x] 用户文档完成
- [x] 所有 QA 证据已收集
- [x] 无残留错误或警告
- [x] 用户批准完整集成
