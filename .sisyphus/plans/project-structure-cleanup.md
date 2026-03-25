# 项目结构清理计划

**创建时间**: 2026-03-25
**目标**: 清理项目中不规范的文件位置和不必要的跟踪文件

---

## 任务清单

### 任务 1: 移动文档文件到正确位置

**文件**: `docs/更新记录_评估指标扩展_2026-03-25_16-27.md`
**目标**: `docs/reports/更新记录_评估指标扩展_2026-03-25_16-27.md`

**步骤**:
1. 移动文件到 `docs/reports/` 目录
2. 更新 git 跟踪状态

**命令**:
```bash
git mv "docs/更新记录_评估指标扩展_2026-03-25_16-27.md" "docs/reports/"
```

---

### 任务 2: 清理模型检查点中间产物

**目标目录**: `models/intent_classifier/checkpoint-*`
**保留**: `models/intent_classifier/` 主目录下的配置文件

**要删除的检查点目录**:
- `checkpoint-7/`
- `checkpoint-14/`
- `checkpoint-21/`
- `checkpoint-28/`
- `checkpoint-35/`
- `checkpoint-42/`

**步骤**:
1. 从 git 中移除这些检查点跟踪
2. 删除本地目录

**命令**:
```bash
git rm -r models/intent_classifier/checkpoint-7
git rm -r models/intent_classifier/checkpoint-14
git rm -r models/intent_classifier/checkpoint-21
git rm -r models/intent_classifier/checkpoint-28
git rm -r models/intent_classifier/checkpoint-35
git rm -r models/intent_classifier/checkpoint-42
```

---

### 任务 3: 更新 .gitignore

**需要添加的规则**:

```gitignore
# Test evaluation reports (generated files)
tests/evaluation/reports/*.json
!tests/evaluation/reports/.gitkeep

# Model training checkpoints
models/*/checkpoint-*/
```

**步骤**:
1. 将上述规则添加到 `.gitignore` 文件末尾

---

## 执行顺序

1. **任务 1**: 移动文档文件（简单移动操作）
2. **任务 2**: 清理模型检查点（批量删除）
3. **任务 3**: 更新 .gitignore（配置更新）

---

## 验证清单

执行完成后验证：

- [ ] `docs/reports/更新记录_评估指标扩展_2026-03-25_16-27.md` 存在
- [ ] `docs/更新记录_评估指标扩展_2026-03-25_16-27.md` 不存在
- [ ] `models/intent_classifier/checkpoint-*` 目录已删除
- [ ] `.gitignore` 已更新
- [ ] `git status` 显示预期的变更

---

## 风险评估

- **低风险**: 这些操作主要是文件移动和删除
- **备份建议**: 执行前可以创建 git 分支作为备份
- **可逆性**: 所有操作可以通过 git 恢复

---

## 注意事项

- 根目录的 `start_project_prod.bat` 和 `start_project_prod.sh` 保留不动（用户要求）
- `data/` 目录缺失问题需要后续讨论是否创建或更新 README
