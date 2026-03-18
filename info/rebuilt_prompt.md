## Plan: BERT意图替换与全套文档部署交付

在不破坏现有流水线与回退机制的前提下，先将意图识别替换为 CPU 友好的 `distilbert-base-uncased`，并修复 NLU 模型加载生命周期（确保实际启用模型而非长期 fallback）；随后补齐中文技术文档（NLU/KG/NLG + 模块间数据传递）与 Windows/macOS 高可用部署文档，同时新增 production 一键脚本并保留现有脚本。

**Steps**
1. 基线与影响面确认（Phase A）
   - 核对当前意图分类推理链路、训练链路、配置入口与测试覆盖。
   - 明确现状缺陷：`GameEngine` 仅实例化 NLU 模块未调用 `load()`，导致模型不生效。
   - 输出改造约束：保持 `rule_fallback` 兜底、接口不变（`predict()->{"intent","confidence"}`）。

2. DistilBERT 替换方案设计（Phase B，*depends on 1*）
   - 将意图模型默认基座切换为 `distilbert-base-uncased`（训练和运行配置同步）。
   - 定义模型目录规范（如 `models/intent_classifier`），并设计“路径不存在时自动回退规则模式”的行为。
   - 保持 `IntentClassifier` 现有 API，避免影响 `GameEngine.process_turn()` 与 UI debug 展示。

3. NLU 加载生命周期修复（Phase C，*depends on 2*）
   - 在 `GameEngine` 初始化阶段显式加载 `coref`、`intent_clf`、`entity_ext`。
   - 增加初始化日志与失败降级日志，确保可观测性。
   - 在应用入口补充可配置模型路径透传（不硬编码）。

4. 训练与推理配套修订（Phase D，*parallel with 3 after interfaces fixed*）
   - 更新 `training/train_intent.py` 文案与默认参数，确保与 DistilBERT 一致。
   - 更新依赖（`transformers`、`torch`）和 README 中训练/推理说明。
   - 明确 CPU 场景下推荐 batch size、max_length、预估时延与资源需求。

5. 测试增强与回归（Phase E，*depends on 3 and 4*）
   - 增加“模型成功加载”“模型路径失效回退”“predict 输出结构稳定”的单元测试。
   - 更新集成测试，覆盖 `GameEngine` 中 NLU debug 字段在模型加载前后行为。
   - 保证现有 fallback 测试仍可通过。

6. 技术文档产出（Phase F，*parallel with 5 once code interfaces stable*）
   - 输出中文技术路线文档：按 NLU、KG、NLG 三部分介绍目标、输入输出、核心算法、降级策略。
   - 输出模块间信息传递文档：逐 turn 数据流（字段级），包含 `TurnResult`、`nlu_debug`、`kg_summary`、`history`、`options`、`conflicts`。
   - 输出实现索引：关键类/函数与文件映射，便于答辩与维护。

7. 高可用部署与脚本交付（Phase G，*depends on 4 and 6*）
   - 保留现有脚本：`start_project.bat`、`start_project.sh`、`start_project_simplified.bat`。
   - 新增 production 脚本：Windows 与 macOS 各一份（建议 `start_project_prod.bat` 与 `start_project_prod.sh`）。
   - production 脚本能力：端口占用检测、进程识别与安全重启、依赖安装超时、环境变量校验、启动失败退出码、日志落盘。
   - 输出双端部署文档：开发模式 vs 生产模式、首次部署、升级、故障排查、回滚。

8. 交付验收与边界确认（Phase H，*depends on 5, 6, 7*）
   - 验证本地一键启动（Windows/macOS）均能完成依赖安装、模型加载、应用启动。
   - 验证 NLU 模型加载状态在 UI/日志可见。
   - 整理“已完成/未纳入范围/后续建议”并提交。

**Relevant files**
- `src/nlu/intent_classifier.py` — 意图分类模型加载、推理分支、fallback 逻辑。
- `src/engine/game_engine.py` — NLU 模块初始化与加载时机、turn 流水线调用。
- `app.py` — 引擎初始化入口、可选模型路径配置透传、NLU debug 展示。
- `training/train_intent.py` — DistilBERT 训练入口、数据与输出路径。
- `config.py` — 默认模型名、相关运行配置。
- `tests/test_nlu.py` — 意图模块单测扩展（模型加载/回退/结构稳定）。
- `tests/test_integration.py` — 引擎级 NLU 行为回归。
- `requirements.txt` — 训练/推理依赖补齐与版本策略。
- `README.md` — 更新快速开始、训练与部署说明。
- `start_project.bat` — 现有 Windows 一键脚本（保留）。
- `start_project.sh` — 现有 macOS/Linux 一键脚本（保留）。
- `start_project_simplified.bat` — 现有简化脚本（保留并标注用途）。
- `start_project_prod.bat` — 新增 Windows production 启动脚本。
- `start_project_prod.sh` — 新增 macOS production 启动脚本。
- `docs/technical_route_zh.md` — 新增中文技术路线与模块设计文档。
- `docs/data_flow_zh.md` — 新增模块间字段级数据流文档。
- `docs/deployment_windows_zh.md` — 新增 Windows 高可用部署文档。
- `docs/deployment_macos_zh.md` — 新增 macOS 高可用部署文档。

**Verification**
1. 训练验证
   - 运行意图训练脚本，确认可成功保存 DistilBERT checkpoint 到约定目录。
2. 推理验证
   - 启动应用后执行多轮输入，确认 `nlu_debug.intent` 与 `confidence` 正常输出。
   - 模型目录不可用时，确认自动回退 `rule_fallback` 且服务不中断。
3. 测试验证
   - 运行 `pytest tests/test_nlu.py -v`。
   - 运行 `pytest tests/test_integration.py -v`。
4. 部署验证
   - Windows: 执行现有脚本与 production 脚本各 1 次，验证端口检测、重启与退出码。
   - macOS: 执行现有脚本与 production 脚本各 1 次，验证端口检测、重启与退出码。
5. 文档验收
   - 抽样按文档从零部署一遍，确认步骤无缺漏、命令可直接执行。

**Decisions**
- 模型选型：`distilbert-base-uncased`（英文优先，CPU 友好）。
- 目标环境：CPU 为主。
- 文档语言：中文。
- 脚本策略：新增 production 脚本，同时保留现有脚本。
- 兼容原则：保留当前 fallback 能力，不移除规则路径。

**Further Considerations**
1. 模型工件分发方式
   - 推荐 A：仓库不提交大模型，仅文档说明下载/训练后路径。
   - 备选 B：提交轻量 checkpoint（若体积允许）。
2. 生产日志策略
   - 推荐 A：脚本按日期写入 `logs/` 并保留最近 N 份。
   - 备选 B：仅控制台日志（实现简单）。
3. macOS 支持边界
   - 推荐 A：文档标注脚本兼容 Linux/macOS（当前 `start_project.sh` 结构已如此）。
   - 备选 B：拆分 macOS 与 Linux 两份脚本（维护成本更高）。
