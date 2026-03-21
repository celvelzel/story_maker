# StoryWeaver 模块间数据传递（字段级）

## 1. 单轮数据流总览

每轮 `process_turn(player_input)` 的数据流如下：

1. 输入层：玩家输入 `player_input`
2. NLU 层：得到 `resolved_input`、`intent`、`confidence`、`entities`
3. NLG 层：根据 NLU + `kg_summary` + `history` 生成 `story_text`
4. KG 层：更新实体关系，输出 `kg_summary`、`kg_html`、`conflicts`
5. UI 层：渲染故事文本、分支选项、NLU debug、KG 图

## 2. 核心结构

### 2.1 TurnResult

字段说明：

1. `story_text: str`：本轮叙事输出
2. `options: List[StoryOption]`：可选行动
3. `nlu_debug: Dict`：NLU 调试信息
4. `kg_html: str`：图谱可视化 HTML
5. `conflicts: List[str]`：一致性冲突文本

### 2.2 nlu_debug

常见字段：

1. `resolved_input`
2. `intent`
3. `confidence`
4. `entities`
5. `intent_backend`
6. `intent_model_loaded`
7. `coref_loaded`
8. `entity_model_loaded`

### 2.3 kg_summary

来源：`KnowledgeGraph.to_summary()`

用途：

1. 为 story_generator 提供世界状态摘要
2. 为 option_generator 提供上下文约束

### 2.4 history

来源：`GameState.recent_history(window)`

用途：

1. 提供近轮上下文
2. 保证叙事连贯与角色一致性

## 3. 逐 turn 字段流转

1. 输入：`player_input`
2. 经过 coref：`resolved_input`
3. 意图分类：`intent`, `confidence`
4. 实体识别：`entities`
5. 主状态写入：`state.story_history` 追加 player
6. 文本生成：`story_text`
7. 状态写入：`state.story_history` 追加 narrator
8. KG 增量：
   - `entities` 先写入
   - `story_text` 抽取 relations 再写入
9. 冲突检测：`conflicts`
10. 选项生成：`options`
11. 结果封装：`TurnResult`

## 4. 实现索引（关键文件）

1. `src/engine/game_engine.py`
2. `src/engine/state.py`
3. `src/nlu/intent_classifier.py`
4. `src/nlu/coreference.py`
5. `src/nlu/entity_extractor.py`
6. `src/knowledge_graph/graph.py`
7. `src/knowledge_graph/relation_extractor.py`
8. `src/knowledge_graph/conflict_detector.py`
9. `src/nlg/story_generator.py`
10. `src/nlg/option_generator.py`
11. `app.py`
