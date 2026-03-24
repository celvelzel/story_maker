# Main Branch Integration Plan: UI Freeze + Performance Gains + KG Quality Recovery

## Plan Metadata
- Project: `story_maker`
- Date: 2026-03-24
- Planner: Prometheus
- Source draft: `.sisyphus/drafts/branch-integration-frontend-ux-vs-nlu-kg.md`

## Objective
在不破坏已定稿前端 UI 的前提下，将 `feat/frontend-ux-optimization` 与 `feat/nlu-kg-quality-first-wave` 的优势有序集成到 `main`：
1) 保持定稿 UI 不回退；
2) 引入已验证的流畅性相关改进；
3) 修复并稳定知识图谱显示质量。

## Locked Decisions (User Confirmed)
1. `nlu-kg` 引入方式：**分阶段选择性引入**（selective cherry-pick/manual port）。
2. 执行顺序：**先 frontend（剩余提交）再 nlu-kg**。
3. 验收优先级：**UI定稿不变 > 性能 > KG显示质量**。
4. 数据策略：允许按 `nlu-kg` 分支原样引入 data 删除相关提交。

## Repository Truth (Exploration-backed)
1. `feat/frontend-ux-optimization` 已并入 `main`（merge commit `1f8b508`），但分支仍有后续提交：`9d8ee9b`、`2d15355`。
2. `feat/nlu-kg-quality-first-wave` 为高侵入分支（约 66 files, +6006/-2011）。
3. 主要冲突热点：`app.py`、`src/engine/game_engine.py`、`src/knowledge_graph/*`、`tests/*`。
4. 可用验证基础设施已存在：
   - CI smoke: `.github/workflows/frontend-smoke.yml`
   - 性能测试：`tests/performance/test_turn_latency.py`
   - KG校验：`tests/kg/*`
   - 质量门禁：`tests/evaluation/test_quality_gates.py` 与 `tests/evaluation/quality_runner.py`

## Scope Boundaries

### IN Scope
- 以“提交白名单”方式将两分支优势引入 `main`。
- 建立分阶段回滚锚点（RP0/CP1/CP2/CP3/CP4）。
- 在每阶段执行 UI 不变性、性能、KG 质量验证。
- 引入用户允许的数据删除提交，并做依赖与回归兜底。

### OUT Scope
- 直接整分支 merge（frontend / nlu-kg）。
- 对现有定稿 UI 做新功能扩展或风格改写。
- 借集成之名进行额外重构（与本次目标无关的代码清理）。
- 在未完成阶段性验证前推进下一阶段。

## Hard Guardrails
1. **禁止**直接 `merge feat/frontend-ux-optimization` 或 `merge feat/nlu-kg-quality-first-wave`。
2. 前端冻结后（CP1）禁止非授权 UI 结构变更。
3. 每阶段必须通过对应 QA 场景后才可进入下一阶段。
4. 任一关键门禁失败，立即回滚到最近 checkpoint，重新切片引入。

## Defaults Applied (Ambiguity Resolved)
1. **数据删除提交 `703b178` 纳入同一集成波次**（Wave 4），但晚于 CP3 且需通过依赖扫描。
2. **性能阈值默认采用相对基线 ≤ 5% 退化**；若更优则记录收益证据。
3. `3fe63e7` 采取条件引入策略：不满足 UI 冻结约束则默认跳过并记录替代覆盖。

## Outcome-level Acceptance Criteria
1. **UI Invariance (P0)**: 前端 smoke + `tests/ui/scenarios` 在 CP1/CP2/CP3/CP4/main 全部通过，且无未授权 UI 结构漂移。
2. **Performance (P1)**: `tests/performance/test_turn_latency.py` 通过，且相对 RP0 基线退化不超过 5%。
3. **KG Quality (P2)**: `tests/kg/*` + `tests/evaluation/test_quality_gates.py` + quality compare 全部通过。
4. **Rollback Readiness**: 任一 CP 失败可在单步内回退到前一 checkpoint 并重跑门禁。

## Commit Selection Policy

### Frontend Branch (allowlist)
- 必须引入：`9d8ee9b`、`2d15355`

### NLU/KG Branch (allowlist, dependency order)
- 门禁与质量基础：`1e40ef1`、`a5fd0ba`
- 条件引入：`3fe63e7`（仅在不破坏现有 UI 测试假设时引入）
- 核心能力序列：`0476a19` → `b57753a` → `6aff223` → `20b60f9`
- 运行时清理能力：`ebd874a`（优先 manual port 到当前 `app.py`）
- 数据删除：`703b178`（置于后段，且需先通过依赖扫描与回归）

### NLU/KG Branch (denylist by default)
- 默认不引入（除非用户后续显式授权）：`6186ad3`、`e388b26`、`ea53582`、`2eaf78c`
  - 理由：这些提交涉及 UI 模块化/结构性重排，易与“UI 定稿不变”冲突。

## Rollback & Checkpoint Strategy
- **RP0**: `main` 当前基线（建议标记 `rp0-main-8f6cbbc`）。
- **CP1**: frontend 后续提交引入并通过 UI 冻结验证。
- **CP2**: 门禁/质量基础能力引入并通过 smoke+测试。
- **CP3**: NLU/KG 核心能力序列完成并通过性能+KG测试。
- **CP4**: 运行时清理与数据删除（若启用）完成并通过全量门禁。

## Detailed Execution Tasks

### Wave 0 — Baseline & Preflight (No Functional Changes)

#### Task W0-1 — Create Integration Branch + Baseline Anchor
- **Goal**: 在任何集成前锁定可一键回退基线。
- **Execution**:
  1. 从 `main` 最新提交创建集成分支（示例：`integration/ui-lock-perf-kg-wave`）。
  2. 标记 `RP0` 标签（示例：`rp0-main-8f6cbbc`，实际以当前 main tip 为准）。
  3. 记录对比基线：`main` 与两个 feature 分支的差异摘要（仅记录，不修改代码）。
- **QA Scenarios (must pass)**:
  - `git status` 干净。
  - `RP0` 标签存在且可指向当前 main 基线。
  - 基线记录包含：frontend 剩余 2 提交与 nlu-kg allowlist/denylist。

#### Task W0-2 — Lock Verification Command Contract
- **Goal**: 固化全流程验证命令，避免阶段间口径漂移。
- **Execution**:
  1. 固定 UI 不变性命令：CI smoke 对应命令 + `tests/ui/scenarios`。
  2. 固定性能命令：`tests/performance/test_turn_latency.py`。
  3. 固定 KG 命令：`tests/kg/*` + `tests/evaluation/test_quality_gates.py` + quality runner compare。
  4. 设定统一阈值：性能相对基线退化不得超过 5%。
- **QA Scenarios (must pass)**:
  - 每类门禁都有明确命令和通过标准（PASS/FAIL）。
  - 质量门禁命令在当前分支可执行（至少 dry-run 校验入口无误）。

#### Task W0-3 — Capture Baseline Evidence Bundle
- **Goal**: 为后续“是否回退/继续”提供客观证据。
- **Execution**:
  1. 在 RP0 上运行 UI/性能/KG 基线测试并保存结果快照。
  2. 对关键 UI 路径记录“定稿不变性”证据（场景级，不要求视觉重设计）。
  3. 输出基线摘要（提交到执行报告，不进入业务源码）。
- **QA Scenarios (must pass)**:
  - 三类基线证据齐全：UI、性能、KG。
  - 基线产物可用于后续阶段同比对。

### Wave 1 — Frontend Post-Merge Delta Integration (UI Lock Phase)

#### Task W1-1 — Integrate `9d8ee9b` via Cherry-pick Only
- **Goal**: 引入 frontend 分支中 main 尚未包含的关键后续修正。
- **Execution**:
  1. 仅 cherry-pick `9d8ee9b`（禁止 branch merge）。
  2. 冲突仅按“保持定稿 UI”原则处理，禁止顺带重构。
  3. 记录冲突与处理决策（文件级）。
- **QA Scenarios (must pass)**:
  - UI smoke 与 `tests/ui/scenarios` 全绿。
  - `app.py` 与 `src/ui/*` 未出现非目标型结构漂移。
  - 工作区干净，提交边界清晰（单一目标）。

#### Task W1-2 — Integrate `2d15355` via Cherry-pick Only
- **Goal**: 完成 frontend 后续剩余提交引入，形成 UI 冻结基准。
- **Execution**:
  1. 仅 cherry-pick `2d15355`。
  2. 若与前一步冲突，按“最小变更”策略修复。
  3. 更新阶段日志并准备 CP1 评审。
- **QA Scenarios (must pass)**:
  - 与 W1-1 相同门禁再次通过。
  - 侧边栏路径场景测试通过（含 KG 区域可达性）。
  - 无新增 UI 组件/布局重排。

#### Task W1-3 — Establish CP1 UI Freeze Checkpoint
- **Goal**: 锁定 UI 不变性阶段成果，后续仅允许非 UI 目标改动。
- **Execution**:
  1. 标记 `CP1` 标签。
  2. 生成“UI 冻结清单”：关键界面结构、交互路径、不可变项。
  3. 为后续每波次增加“UI 漂移审计”步骤。
- **QA Scenarios (must pass)**:
  - CP1 标签存在且对应通过全部 UI 门禁的提交。
  - UI 冻结清单可用于后续逐波审计。

### Wave 2 — Gate Infrastructure Import (Before Core NLU/KG Changes)

#### Task W2-1 — Import CI/Smoke Gate Commit `1e40ef1`
- **Goal**: 在引入核心能力前先统一验证通道。
- **Execution**:
  1. cherry-pick `1e40ef1`。
  2. 校验 workflow 与现有目录结构、测试路径兼容。
  3. 如遇路径偏差，仅做最小兼容修正，不变更测试语义。
- **QA Scenarios (must pass)**:
  - workflow 可被触发（本地/CI 入口有效）。
  - 前端 smoke 不因路径或配置失效。
  - UI 冻结审计通过（无 UI 结构变化）。

#### Task W2-2 — Import Quality Foundation Commit `a5fd0ba`
- **Goal**: 引入质量评估与性能验证基础设施。
- **Execution**:
  1. cherry-pick `a5fd0ba`。
  2. 对齐 `tests/evaluation/*` 与现有执行命令。
  3. 确认 quality gate 与基线结果可比。
- **QA Scenarios (must pass)**:
  - `tests/evaluation/test_quality_gates.py` 可执行并给出稳定结果。
  - 性能测试入口保持可用。
  - UI 冻结审计继续通过。

#### Task W2-3 — Conditional Import `3fe63e7`
- **Goal**: 在不破坏现有 UI 场景假设时增强 UI smoke 覆盖。
- **Execution**:
  1. 先做 dry-run 级冲突评估（不落地时不推进）。
  2. 仅当其测试语义与“UI 定稿不变”一致时 cherry-pick。
  3. 若会引入 UI 结构改写假设，则跳过并记录原因。
- **QA Scenarios (must pass)**:
  - 引入后 UI 场景测试稳定，无新 flaky。
  - 若跳过，报告中有明确替代验证策略与覆盖说明。

#### Task W2-4 — Establish CP2 Gate-Ready Checkpoint
- **Goal**: 锁定“可持续验证”能力后再进入高风险核心变更。
- **Execution**:
  1. 标记 `CP2` 标签。
  2. 归档 CP1→CP2 期间所有门禁结果。
  3. 通过后方可解锁 Wave 3。
- **QA Scenarios (must pass)**:
  - 三类门禁均可运行且结果可追溯。
  - CP2 标签对应提交具备完整验证证据。

### Wave 3 — Core NLU/KG Selective Integration (Dependency Ordered)

#### Task W3-1 — Import KG Strategy Toggle Base `0476a19`
- **Goal**: 建立后续 KG/性能改动所依赖的配置地基。
- **Execution**:
  1. cherry-pick `0476a19`。
  2. 校验配置项与当前 `config.py`/runtime 读取路径一致。
  3. 若存在默认值冲突，采用“保持当前 main 兼容 + 新开关显式默认”策略。
- **QA Scenarios (must pass)**:
  - 启动流程不因新配置键报错。
  - KG 相关基础测试可运行。
  - UI 冻结审计通过。

#### Task W3-2 — Import Entity Alias/Confidence Layer `b57753a`
- **Goal**: 引入 KG/NLU 实体识别质量提升基础。
- **Execution**:
  1. cherry-pick `b57753a`。
  2. 对齐实体别名归一化与置信度输出接口，避免破坏现有消费方。
  3. 必要时增加兼容适配层（仅限接口映射，不改 UI 外观）。
- **QA Scenarios (must pass)**:
  - `tests/nlu/*` 与 `tests/kg/*` 中相关用例通过。
  - KG 显示数据结构无断裂（面板可渲染，字段完整）。
  - 性能较基线无显著回退（预检）。

#### Task W3-3 — Import Conflict/Quarantine Hardening `6aff223`
- **Goal**: 降低 KG 冲突与错误关系进入展示层的概率。
- **Execution**:
  1. cherry-pick `6aff223`。
  2. 处理 `src/knowledge_graph/*` 与 `src/engine/game_engine.py` 冲突时优先保留 CP1 UI 冻结约束。
  3. 对冲突隔离逻辑添加回归测试映射（引用现有 tests/kg）。
- **QA Scenarios (must pass)**:
  - `tests/kg/test_conflict_resolution.py` 与相关时序/类型测试通过。
  - KG 面板无“空白/崩溃/字段错位”回归。
  - UI 冻结审计继续通过。

#### Task W3-4 — Import Incremental Importance/Summary Caching `20b60f9`
- **Goal**: 引入性能收益核心改动并保持 KG 质量稳定。
- **Execution**:
  1. cherry-pick `20b60f9`。
  2. 校验与 `0476a19` 的配置依赖和默认行为一致。
  3. 对缓存与增量逻辑建立回归对照（与 RP0/CP1 基线比较）。
- **QA Scenarios (must pass)**:
  - `tests/performance/test_turn_latency.py` 通过。
  - 性能相对基线退化 ≤ 5%（若有提升需记录证据）。
  - KG 质量门禁（`tests/kg/*` + evaluation gate）通过。

#### Task W3-5 — Establish CP3 Core Integration Checkpoint
- **Goal**: 锁定核心能力引入结果，隔离后续运行时与数据操作风险。
- **Execution**:
  1. 标记 `CP3` 标签。
  2. 归档 W3 全部冲突处理与验证记录。
  3. 若任一门禁未过，回退到 CP2 重新切片，不允许带病进入 Wave 4。
- **QA Scenarios (must pass)**:
  - CP3 证据包包含 UI/性能/KG 三类通过记录。
  - 回退路径（CP3→CP2）在演练中可执行。

### Wave 4 — Runtime Stability + Data Deletion Introduction

#### Task W4-1 — Manual Port Runtime Cleanup from `ebd874a`
- **Goal**: 引入运行时清理收益，同时避免覆盖当前 `app.py` 的 UI 冻结结构。
- **Execution**:
  1. 不直接 cherry-pick 全提交，优先 manual port 到当前 `app.py`。
  2. 明确排除 `.sisyphus/boulder.json` 等非运行必需文件。
  3. 仅落地与 signal/runtime cleanup 直接相关逻辑。
- **QA Scenarios (must pass)**:
  - 应用启动/退出流程稳定，无重复清理或资源泄漏异常。
  - UI 结构与交互路径无变化。
  - 关键回归测试通过。

#### Task W4-2 — Introduce Data Deletion Commit `703b178` (User-approved)
- **Goal**: 按用户授权引入 data 删除提交，并验证无隐性依赖断裂。
- **Execution**:
  1. 在 CP3 基础上引入 `703b178`。
  2. 执行“悬挂引用”扫描：检查 scripts/tests/runtime 是否仍引用被删数据路径。
  3. 对必要流程补充迁移说明（文档或执行说明，不做额外功能开发）。
- **QA Scenarios (must pass)**:
  - 测试与运行脚本不因缺失数据路径崩溃。
  - 评估流程入口仍可执行（允许结果变化，但不允许流程中断）。
  - UI 冻结与性能门禁再次通过。

#### Task W4-3 — Establish CP4 Cutover Candidate Checkpoint
- **Goal**: 形成“可进入 main”的候选状态。
- **Execution**:
  1. 标记 `CP4` 标签。
  2. 产出 CP3→CP4 变更摘要（含数据删除影响面）。
  3. 准备最终验证波次材料。
- **QA Scenarios (must pass)**:
  - CP4 标签与证据包一致。
  - 所有高优先门禁（UI优先）已在 CP4 上复核通过。

### Wave 5 — Main Cutover

#### Task W5-1 — Final Rebase/Conflict Audit Against Latest `main`
- **Goal**: 在切入 `main` 前清理最后的漂移风险。
- **Execution**:
  1. 同步最新 `main`，执行最小化 rebase/merge（按团队规范）。
  2. 重新执行冲突审计，确保未引入 denylist 变更。
  3. 更新最终变更清单（提交级 + 文件级）。
- **QA Scenarios (must pass)**:
  - 无未解释冲突残留。
  - allowlist 全部命中、denylist 未误入。
  - UI 冻结审计继续通过。

#### Task W5-2 — Merge to `main` with Evidence-linked PR
- **Goal**: 将 CP4 候选安全并入 `main`。
- **Execution**:
  1. 提交 PR，正文包含 RP0~CP4 的门禁证据链接。
  2. 标注默认决策（性能阈值、条件跳过 `3fe63e7` 等）与理由。
  3. 合并后立即执行 main 分支复核验证。
- **QA Scenarios (must pass)**:
  - PR 审核信息完整且可追溯。
  - 合并后 main 上三大门禁复跑通过。
  - 若失败，按最近 checkpoint 回退，不做临场热修混入。

## Final Verification Wave (Explicit User "okay" Required)
1. 产出集成验证报告（UI不变性/性能/KG质量三大门禁 + 证据链接）。
2. 列出所有默认决策与自动修复项（含风险说明与回滚点）。
3. 明确 residual risk（若有）及对应后续任务建议。
4. **停止并请求用户明确回复“okay”后，方可标记本次集成执行完成。**

## Handoff
- 执行入口：`/start-work`
- 高准确模式：执行前可触发 Momus 审核循环。
