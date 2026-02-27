# StoryWeaver 混合架构实现计划

> **项目**: COMP5423 NLP — Interactive Text Adventure Story Generator  
> **架构**: 混合方案 (本地 NLU + API NLG)  
> **最后更新**: 2026-02-27

---

## 一、架构总览

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────┐
│  NLU Pipeline (本地)                              │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Intent      │  │ Entity   │  │ Coreference │ │
│  │ Classifier  │  │ Extractor│  │ Resolver    │ │
│  │ (RoBERTa)   │  │ (spaCy)  │  │ (fastcoref) │ │
│  └──────┬──────┘  └────┬─────┘  └──────┬──────┘ │
│         └───────┬──────┘───────────────┘         │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  Knowledge Graph (本地 NetworkX)                  │
│  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ Graph Manager  │  │ Conflict Detector       │ │
│  │ (MultiDiGraph) │  │ (Rules + LLM fallback)  │ │
│  └───────┬───────┘  └───────────┬──────────────┘ │
│          └──────────┬───────────┘                 │
└─────────────────────┼────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│  NLG Pipeline (OpenAI API)                        │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ Story        │  │ Option       │               │
│  │ Generator    │  │ Generator    │               │
│  │ (gpt-4o-mini)│  │ (gpt-4o-mini)│               │
│  └──────┬──────┘  └──────┬───────┘               │
│         └───────┬────────┘                        │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  Gradio UI                                        │
│  Chat + Option Buttons + KG Visualization (PyVis) │
└──────────────────────────────────────────────────┘
```

---

## 二、目录结构

```
story_maker/
├── .env                          # API keys (git-ignored)
├── .env.example                  # 示例环境变量
├── .gitignore
├── config.py                     # 全局配置 (Pydantic Settings)
├── app.py                        # Gradio 前端入口
├── requirements.txt              # 依赖清单
├── README.md
│
├── info/                         # 项目文档
│   ├── implementation_plan.md    # 本文件
│   ├── agent_prompt.md           # Agent 提示词
│   └── *.pdf                     # 课程项目说明
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── api_client.py         # 统一 LLM API 客户端
│   │
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # RoBERTa 意图分类
│   │   ├── entity_extractor.py   # spaCy NER + 名词短语
│   │   └── coreference.py        # fastcoref 共指消解
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph.py              # NetworkX MultiDiGraph 管理
│   │   ├── relation_extractor.py # LLM 关系抽取
│   │   ├── conflict_detector.py  # 规则 + LLM 冲突检测
│   │   └── visualizer.py         # PyVis 可视化
│   │
│   ├── nlg/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py   # 版本化 Prompt 模板
│   │   ├── story_generator.py    # 故事续写 (gpt-4o-mini)
│   │   └── option_generator.py   # 选项生成 (gpt-4o-mini)
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py        # 游戏主循环编排
│   │   └── state.py              # GameState 数据类
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # Distinct-n, Self-BLEU 等
│       └── llm_judge.py          # LLM-as-Judge 评估
│
├── tests/
│   ├── __init__.py
│   ├── test_nlu.py
│   ├── test_knowledge_graph.py
│   ├── test_nlg.py
│   ├── test_engine.py
│   └── test_integration.py
│
├── data/
│   ├── intent_labels.json        # 意图标签映射
│   └── scripts/
│       └── download_data.py
│
└── training/
    └── train_intent.py           # 意图分类器微调脚本
```

---

## 三、模块详细设计

### 3.1 config.py — 全局配置

```python
"""Global configuration using pydantic-settings."""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # --- API ---
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.8
    LLM_MAX_TOKENS: int = 512
    
    # --- NLU ---
    INTENT_MODEL_NAME: str = "roberta-base"
    INTENT_LABELS: list[str] = [
        "action", "dialogue", "explore",
        "use_item", "ask_info", "rest", "trade", "other"
    ]
    SPACY_MODEL: str = "en_core_web_sm"
    
    # --- Knowledge Graph ---
    KG_MAX_NODES: int = 200
    KG_ENTITY_TYPES: list[str] = ["person", "location", "item", "creature"]
    KG_RELATION_TYPES: list[str] = [
        "located_at", "owns", "knows",
        "enemy_of", "ally_of", "caused", "contains"
    ]
    
    # --- Game ---
    MAX_HISTORY_TURNS: int = 10
    NUM_OPTIONS: int = 3
    
    # --- Evaluation ---
    EVAL_DIMENSIONS: list[str] = [
        "narrative_quality", "consistency",
        "player_agency", "creativity", "pacing"
    ]
    
    # --- Gradio ---
    GRADIO_SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

**要点**:
- 使用 `pydantic-settings` 自动从 `.env` 读取环境变量
- 所有配置集中管理，方便团队统一修改
- `INTENT_LABELS` 8 个游戏内意图：`action, dialogue, explore, use_item, ask_info, rest, trade, other`

---

### 3.2 src/utils/api_client.py — 统一 LLM API 客户端

```python
"""Unified LLM API client with retry, rate-limiting, and cost tracking."""
import time, logging, json
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """Singleton wrapper around OpenAI chat completions."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance
    
    def _init_client(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        # gpt-4o-mini pricing (per 1M tokens)
        self.input_price = 0.15   # $0.15/1M input tokens
        self.output_price = 0.60  # $0.60/1M output tokens
    
    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> str:
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        kwargs = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._track_usage(resp.usage)
                return resp.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
        
        raise RuntimeError("LLM API call failed after max retries")
    
    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        raw = self.chat(messages, json_mode=True, **kwargs)
        return json.loads(raw)
    
    def _track_usage(self, usage):
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.total_cost += (
                usage.prompt_tokens * self.input_price / 1_000_000
                + usage.completion_tokens * self.output_price / 1_000_000
            )
    
    def get_cost_summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }

llm_client = LLMClient()
```

**要点**:
- 单例模式，全局复用
- 指数退避重试 (3 次)
- JSON mode 支持 (`chat_json`)
- 自动统计 token 消耗和费用
- 费用估算：~$0.0003/turn, ~$0.009/game (30 turns)

---

### 3.3 NLU 模块

#### 3.3.1 intent_classifier.py

```python
"""Intent classification using fine-tuned RoBERTa."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import settings

class IntentClassifier:
    def __init__(self, model_path: str | None = None):
        self.labels = settings.INTENT_LABELS
        self.model_path = model_path or settings.INTENT_MODEL_NAME
        self.tokenizer = None
        self.model = None
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=len(self.labels)
        )
        self.model.eval()
    
    def predict(self, text: str) -> dict:
        if self.model is None:
            self.load()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        idx = probs.argmax().item()
        return {
            "intent": self.labels[idx],
            "confidence": probs[idx].item(),
            "all_probs": {l: p.item() for l, p in zip(self.labels, probs)},
        }
    
    def rule_fallback(self, text: str) -> str:
        """Keyword-based fallback when model is not available."""
        text_lower = text.lower()
        rules = {
            "action": ["attack", "fight", "hit", "strike", "kick", "push"],
            "dialogue": ["say", "tell", "ask", "speak", "talk", "greet"],
            "explore": ["look", "examine", "search", "go", "walk", "enter", "move"],
            "use_item": ["use", "equip", "drink", "eat", "open", "throw"],
            "ask_info": ["what", "where", "who", "how", "why", "describe"],
            "rest": ["rest", "sleep", "wait", "camp", "heal"],
            "trade": ["buy", "sell", "trade", "barter", "shop"],
        }
        for intent, keywords in rules.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return "other"
```

**训练数据构造**:
- 每个意图 200-500 条合成训练样本 (可用 GPT 批量生成)
- 训练脚本: `training/train_intent.py`
- 验证集 20%，目标 accuracy ≥ 85%

#### 3.3.2 entity_extractor.py

```python
"""Entity extraction using spaCy NER + noun phrase extraction."""
import spacy
from config import settings

class EntityExtractor:
    # Map spaCy NER labels to game entity types
    LABEL_MAP = {
        "PERSON": "person",
        "GPE": "location",
        "LOC": "location",
        "FAC": "location",
        "ORG": "location",
    }
    
    def __init__(self):
        self.nlp = spacy.load(settings.SPACY_MODEL)
    
    def extract(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        entities = []
        seen = set()
        
        # 1) spaCy NER entities
        for ent in doc.ents:
            key = ent.text.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "type": self.LABEL_MAP.get(ent.label_, "item"),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "ner",
                })
        
        # 2) Noun phrase extraction (game-relevant supplements)
        for chunk in doc.noun_chunks:
            key = chunk.text.lower()
            if key not in seen and len(chunk.text.split()) <= 3:
                if chunk.root.pos_ in ("NOUN", "PROPN"):
                    seen.add(key)
                    entities.append({
                        "text": chunk.text,
                        "type": self._infer_type(chunk.root),
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "source": "noun_phrase",
                    })
        
        return entities
    
    def _infer_type(self, token) -> str:
        if token.pos_ == "PROPN":
            return "person"
        creature_words = {"dragon", "wolf", "goblin", "troll", "monster", "beast"}
        if token.text.lower() in creature_words:
            return "creature"
        location_words = {"cave", "forest", "castle", "village", "river", "mountain", "tower"}
        if token.text.lower() in location_words:
            return "location"
        return "item"
```

#### 3.3.3 coreference.py

```python
"""Coreference resolution using fastcoref."""
from fastcoref import FCoref

class CoreferenceResolver:
    def __init__(self):
        self.model = None
    
    def load(self):
        self.model = FCoref(device="cpu")
    
    def resolve(self, text: str) -> str:
        if self.model is None:
            self.load()
        preds = self.model.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=True)
        resolved = text
        # Simple replacement: replace pronouns with first mention
        for cluster in clusters:
            if len(cluster) >= 2:
                main = cluster[0]
                for mention in cluster[1:]:
                    if mention.lower() in ("he", "she", "it", "they", "him", "her", "them"):
                        resolved = resolved.replace(mention, main, 1)
        return resolved
```

---

### 3.4 Knowledge Graph 模块

#### 3.4.1 graph.py — MultiDiGraph 管理

```python
"""Knowledge graph manager using NetworkX MultiDiGraph."""
import networkx as nx
from typing import Optional
from config import settings

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_entity(self, name: str, entity_type: str, **attrs):
        name_key = name.lower()
        if self.graph.has_node(name_key):
            self.graph.nodes[name_key].update(attrs)
        else:
            self.graph.add_node(name_key, name=name, type=entity_type, **attrs)
        self._enforce_limit()
    
    def add_relation(self, source: str, target: str, relation: str, **attrs):
        s, t = source.lower(), target.lower()
        if not self.graph.has_node(s):
            self.add_entity(source, "unknown")
        if not self.graph.has_node(t):
            self.add_entity(target, "unknown")
        # Avoid duplicate edges with same relation
        existing = self.graph.get_edge_data(s, t)
        if existing:
            for key, data in existing.items():
                if data.get("relation") == relation:
                    self.graph.edges[s, t, key].update(attrs)
                    return
        self.graph.add_edge(s, t, relation=relation, **attrs)
    
    def get_entity(self, name: str) -> Optional[dict]:
        name_key = name.lower()
        if self.graph.has_node(name_key):
            return dict(self.graph.nodes[name_key])
        return None
    
    def get_relations(self, entity: str) -> list[dict]:
        name_key = entity.lower()
        results = []
        if not self.graph.has_node(name_key):
            return results
        for _, target, data in self.graph.out_edges(name_key, data=True):
            results.append({"source": name_key, "target": target, **data})
        for source, _, data in self.graph.in_edges(name_key, data=True):
            results.append({"source": source, "target": name_key, **data})
        return results
    
    def remove_entity(self, name: str):
        name_key = name.lower()
        if self.graph.has_node(name_key):
            self.graph.remove_node(name_key)
    
    def update_attribute(self, name: str, key: str, value):
        name_key = name.lower()
        if self.graph.has_node(name_key):
            self.graph.nodes[name_key][key] = value
    
    def to_summary(self, max_entities: int = 30) -> str:
        """Convert KG to text summary for LLM prompt injection."""
        lines = []
        nodes = list(self.graph.nodes(data=True))[:max_entities]
        
        lines.append("=== World State ===")
        for node_id, data in nodes:
            name = data.get("name", node_id)
            etype = data.get("type", "unknown")
            attrs = {k: v for k, v in data.items() if k not in ("name", "type")}
            attr_str = f" ({', '.join(f'{k}={v}' for k, v in attrs.items())})" if attrs else ""
            lines.append(f"- {name} [{etype}]{attr_str}")
        
        lines.append("\n=== Relations ===")
        for s, t, data in list(self.graph.edges(data=True))[:50]:
            rel = data.get("relation", "related_to")
            s_name = self.graph.nodes[s].get("name", s)
            t_name = self.graph.nodes[t].get("name", t)
            lines.append(f"- {s_name} --{rel}--> {t_name}")
        
        return "\n".join(lines)
    
    def _enforce_limit(self):
        while self.graph.number_of_nodes() > settings.KG_MAX_NODES:
            # Remove oldest/least connected node
            min_node = min(self.graph.nodes(), key=lambda n: self.graph.degree(n))
            self.graph.remove_node(min_node)
    
    def get_stats(self) -> dict:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
        }
```

#### 3.4.2 relation_extractor.py — LLM 关系抽取

```python
"""LLM-based relation extraction from story text."""
from src.utils.api_client import llm_client
from config import settings

EXTRACTION_PROMPT = """Extract entities and relations from the following story text.

Entity types: {entity_types}
Relation types: {relation_types}

Story text:
{text}

Respond in JSON format:
{{
  "entities": [
    {{"name": "...", "type": "person|location|item|creature", "attributes": {{}}}}
  ],
  "relations": [
    {{"source": "...", "relation": "located_at|owns|knows|enemy_of|ally_of|caused|contains", "target": "..."}}
  ]
}}"""

class RelationExtractor:
    def extract(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": "You are a precise information extraction system for fantasy game narratives."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(
                entity_types=", ".join(settings.KG_ENTITY_TYPES),
                relation_types=", ".join(settings.KG_RELATION_TYPES),
                text=text,
            )},
        ]
        try:
            result = llm_client.chat_json(messages, temperature=0.1, max_tokens=512)
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", []),
            }
        except Exception as e:
            return {"entities": [], "relations": [], "error": str(e)}
```

#### 3.4.3 conflict_detector.py — 双层冲突检测

```python
"""Two-layer conflict detection: rules + LLM reasoning."""
from src.knowledge_graph.graph import KnowledgeGraph
from src.utils.api_client import llm_client

class ConflictDetector:
    # Mutually exclusive relations
    EXCLUSIVE_PAIRS = [
        ("ally_of", "enemy_of"),
        ("alive", "dead"),
    ]
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def check_all(self, new_text: str | None = None) -> list[dict]:
        conflicts = []
        conflicts.extend(self._rule_based_check())
        if new_text:
            conflicts.extend(self._llm_check(new_text))
        return conflicts
    
    def _rule_based_check(self) -> list[dict]:
        conflicts = []
        for s, t, data in self.kg.graph.edges(data=True):
            rel = data.get("relation")
            for r1, r2 in self.EXCLUSIVE_PAIRS:
                if rel == r1:
                    # Check if opposing relation exists
                    edges = self.kg.graph.get_edge_data(s, t)
                    if edges:
                        for _, edata in edges.items():
                            if edata.get("relation") == r2:
                                conflicts.append({
                                    "type": "exclusive_relation",
                                    "entity1": s, "entity2": t,
                                    "relation1": r1, "relation2": r2,
                                    "severity": "high",
                                })
        
        # Attribute conflicts
        for node, data in self.kg.graph.nodes(data=True):
            if data.get("status") == "dead" and data.get("health", 0) > 0:
                conflicts.append({
                    "type": "attribute_conflict",
                    "entity": node,
                    "detail": "Entity is dead but has positive health",
                    "severity": "medium",
                })
        
        return conflicts
    
    def _llm_check(self, new_text: str) -> list[dict]:
        kg_summary = self.kg.to_summary(max_entities=20)
        messages = [
            {"role": "system", "content": "You are a story consistency checker. Identify logical contradictions."},
            {"role": "user", "content": f"""Given the current world state and a new story development, 
identify any logical contradictions or inconsistencies.

{kg_summary}

New development:
{new_text}

Respond in JSON:
{{
  "conflicts": [
    {{"description": "...", "severity": "high|medium|low", "suggestion": "..."}}
  ]
}}
If no conflicts, return {{"conflicts": []}}"""},
        ]
        try:
            result = llm_client.chat_json(messages, temperature=0.1)
            return [
                {**c, "type": "llm_detected"} for c in result.get("conflicts", [])
            ]
        except Exception:
            return []
```

#### 3.4.4 visualizer.py — PyVis 可视化

```python
"""Knowledge graph visualization using PyVis."""
from pyvis.network import Network
from src.knowledge_graph.graph import KnowledgeGraph

# Entity type → color mapping
ENTITY_STYLES = {
    "person": {"color": "#FF6B6B", "shape": "dot"},
    "location": {"color": "#4ECDC4", "shape": "diamond"},
    "item": {"color": "#FFE66D", "shape": "triangle"},
    "creature": {"color": "#A855F7", "shape": "star"},
    "unknown": {"color": "#95A5A6", "shape": "dot"},
}

class KGVisualizer:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def to_html(self, height: str = "500px", width: str = "100%") -> str:
        net = Network(height=height, width=width, directed=True, notebook=False)
        net.barnes_hut(gravity=-3000, spring_length=150)
        
        for node, data in self.kg.graph.nodes(data=True):
            etype = data.get("type", "unknown")
            style = ENTITY_STYLES.get(etype, ENTITY_STYLES["unknown"])
            label = data.get("name", node)
            title = f"{label} [{etype}]"
            # Add attributes to tooltip
            attrs = {k: v for k, v in data.items() if k not in ("name", "type")}
            if attrs:
                title += "\n" + "\n".join(f"  {k}: {v}" for k, v in attrs.items())
            net.add_node(node, label=label, title=title, **style)
        
        for s, t, data in self.kg.graph.edges(data=True):
            rel = data.get("relation", "related")
            net.add_edge(s, t, title=rel, label=rel, arrows="to")
        
        return net.generate_html()
```

---

### 3.5 NLG 模块

#### 3.5.1 prompt_templates.py

```python
"""Versioned prompt templates for LLM story generation."""

SYSTEM_PROMPT = """You are a master storyteller for an interactive fantasy text adventure.

Rules:
1. Write in second person ("You see...", "You feel...")
2. Keep each response to 2-4 paragraphs
3. Maintain consistency with the world state provided
4. Create vivid, immersive descriptions
5. React meaningfully to the player's chosen action
6. Never kill the player without warning
7. Introduce new elements that expand the world organically"""

STORY_CONTINUE_PROMPT = """## World State
{kg_summary}

## Recent History
{history}

## Player Action
Intent: {intent}
Action: "{player_input}"

Continue the story based on the player's action. Be creative but consistent with the world state."""

OPTION_GENERATION_PROMPT = """Based on the current story situation, generate {num_options} distinct action options for the player.

## Current Situation
{current_story}

## World State
{kg_summary}

Respond in JSON format:
{{
  "options": [
    {{"text": "...", "intent_hint": "action|dialogue|explore|use_item|rest|trade", "risk_level": "low|medium|high"}}
  ]
}}

Rules:
- Each option should lead to a meaningfully different outcome
- At least one safe option and one risky option
- Options should feel natural in the story context
- Keep option text concise (1 sentence)"""

OPENING_PROMPT = """Generate an opening scene for a fantasy text adventure game.

Setting: A mysterious world with magic, danger, and wonder.

Requirements:
1. Set the scene with vivid description (2-3 paragraphs)
2. Introduce the protagonist's situation
3. Create a sense of intrigue and possibility
4. End at a natural decision point"""
```

#### 3.5.2 story_generator.py

```python
"""LLM-based story continuation generator."""
from src.utils.api_client import llm_client
from src.nlg.prompt_templates import SYSTEM_PROMPT, STORY_CONTINUE_PROMPT, OPENING_PROMPT

class StoryGenerator:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
    
    def generate_opening(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": OPENING_PROMPT},
        ]
        return llm_client.chat(messages, temperature=0.9, max_tokens=600)
    
    def continue_story(
        self,
        player_input: str,
        intent: str,
        kg_summary: str,
        history: list[str],
    ) -> str:
        history_text = "\n".join(history[-5:]) if history else "(Game just started)"
        
        prompt = STORY_CONTINUE_PROMPT.format(
            kg_summary=kg_summary,
            history=history_text,
            intent=intent,
            player_input=player_input,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return llm_client.chat(messages)
```

#### 3.5.3 option_generator.py

```python
"""LLM-based option generation for player choices."""
from dataclasses import dataclass
from src.utils.api_client import llm_client
from src.nlg.prompt_templates import OPTION_GENERATION_PROMPT
from config import settings

@dataclass
class StoryOption:
    text: str
    intent_hint: str
    risk_level: str

class OptionGenerator:
    def generate(self, current_story: str, kg_summary: str) -> list[StoryOption]:
        prompt = OPTION_GENERATION_PROMPT.format(
            num_options=settings.NUM_OPTIONS,
            current_story=current_story,
            kg_summary=kg_summary,
        )
        messages = [
            {"role": "system", "content": "You are a game design assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            result = llm_client.chat_json(messages, temperature=0.8)
            options = []
            for opt in result.get("options", []):
                options.append(StoryOption(
                    text=opt["text"],
                    intent_hint=opt.get("intent_hint", "other"),
                    risk_level=opt.get("risk_level", "medium"),
                ))
            return options
        except Exception:
            # Fallback options
            return [
                StoryOption("Look around carefully", "explore", "low"),
                StoryOption("Move forward cautiously", "action", "medium"),
                StoryOption("Try to talk to someone nearby", "dialogue", "low"),
            ]
```

---

### 3.6 Engine 模块

#### 3.6.1 state.py

```python
"""Game state management."""
from dataclasses import dataclass, field

@dataclass
class GameState:
    turn: int = 0
    story_history: list[str] = field(default_factory=list)
    player_inputs: list[str] = field(default_factory=list)
    current_scene: str = ""
    is_active: bool = True
    
    def add_turn(self, player_input: str, story_text: str):
        self.turn += 1
        self.player_inputs.append(player_input)
        self.story_history.append(story_text)
        self.current_scene = story_text
    
    def get_recent_history(self, n: int = 5) -> list[str]:
        return self.story_history[-n:]
```

#### 3.6.2 game_engine.py — 核心编排器

```python
"""Game engine: orchestrates NLU → KG → NLG pipeline per turn."""
from dataclasses import dataclass
from src.engine.state import GameState
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.nlu.coreference import CoreferenceResolver
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.relation_extractor import RelationExtractor
from src.knowledge_graph.conflict_detector import ConflictDetector
from src.knowledge_graph.visualizer import KGVisualizer
from src.nlg.story_generator import StoryGenerator
from src.nlg.option_generator import OptionGenerator, StoryOption

@dataclass
class TurnResult:
    story_text: str
    options: list[StoryOption]
    nlu_debug: dict
    kg_html: str
    conflicts: list[dict]

class GameEngine:
    def __init__(self):
        self.state = GameState()
        self.kg = KnowledgeGraph()
        
        # NLU (local)
        self.intent_clf = IntentClassifier()
        self.entity_ext = EntityExtractor()
        self.coref = CoreferenceResolver()
        
        # KG
        self.rel_extractor = RelationExtractor()
        self.conflict_detector = ConflictDetector(self.kg)
        self.visualizer = KGVisualizer(self.kg)
        
        # NLG (API)
        self.story_gen = StoryGenerator()
        self.option_gen = OptionGenerator()
    
    def start_game(self) -> TurnResult:
        opening = self.story_gen.generate_opening()
        self.state.current_scene = opening
        self.state.story_history.append(opening)
        
        # Extract initial world info
        extraction = self.rel_extractor.extract(opening)
        self._apply_extraction(extraction)
        
        options = self.option_gen.generate(opening, self.kg.to_summary())
        
        return TurnResult(
            story_text=opening,
            options=options,
            nlu_debug={"phase": "opening"},
            kg_html=self.visualizer.to_html(),
            conflicts=[],
        )
    
    def process_turn(self, player_input: str) -> TurnResult:
        # 1. Coreference resolution
        resolved_input = self.coref.resolve(player_input)
        
        # 2. NLU: Intent + Entity
        intent_result = self.intent_clf.predict(resolved_input)
        entities = self.entity_ext.extract(resolved_input)
        
        nlu_debug = {
            "original": player_input,
            "resolved": resolved_input,
            "intent": intent_result,
            "entities": entities,
        }
        
        # 3. NLG: Story continuation
        story_text = self.story_gen.continue_story(
            player_input=resolved_input,
            intent=intent_result["intent"],
            kg_summary=self.kg.to_summary(),
            history=self.state.get_recent_history(),
        )
        
        # 4. KG: Extract & update from new story
        combined_text = f"Player: {resolved_input}\n\nStory: {story_text}"
        extraction = self.rel_extractor.extract(combined_text)
        self._apply_extraction(extraction)
        
        # 5. Conflict detection
        conflicts = self.conflict_detector.check_all(new_text=story_text)
        
        # 6. Update state
        self.state.add_turn(player_input, story_text)
        
        # 7. Generate options
        options = self.option_gen.generate(story_text, self.kg.to_summary())
        
        return TurnResult(
            story_text=story_text,
            options=options,
            nlu_debug=nlu_debug,
            kg_html=self.visualizer.to_html(),
            conflicts=conflicts,
        )
    
    def _apply_extraction(self, extraction: dict):
        for ent in extraction.get("entities", []):
            self.kg.add_entity(
                name=ent["name"],
                entity_type=ent.get("type", "unknown"),
                **ent.get("attributes", {}),
            )
        for rel in extraction.get("relations", []):
            self.kg.add_relation(
                source=rel["source"],
                target=rel["target"],
                relation=rel["relation"],
            )
```

---

### 3.7 Evaluation 模块

#### 3.7.1 metrics.py

```python
"""Automated evaluation metrics."""
from collections import Counter
import math

def distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct-n: ratio of unique n-grams to total n-grams."""
    total, unique = 0, set()
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        unique.update(ngrams)
        total += len(ngrams)
    return len(unique) / total if total > 0 else 0.0

def self_bleu(texts: list[str], n: int = 4) -> float:
    """Self-BLEU: average BLEU of each text against all others (lower = more diverse)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    scores = []
    for i, text in enumerate(texts):
        refs = [t.split() for j, t in enumerate(texts) if j != i]
        hyp = text.split()
        if refs and hyp:
            weights = tuple([1.0/n] * n)
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smooth)
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def entity_coverage(kg_entities: set[str], story_texts: list[str]) -> float:
    """Fraction of KG entities mentioned in story texts."""
    if not kg_entities:
        return 0.0
    combined = " ".join(story_texts).lower()
    mentioned = sum(1 for e in kg_entities if e.lower() in combined)
    return mentioned / len(kg_entities)

def consistency_rate(conflicts_per_turn: list[int]) -> float:
    """Fraction of turns with zero conflicts."""
    if not conflicts_per_turn:
        return 1.0
    clean = sum(1 for c in conflicts_per_turn if c == 0)
    return clean / len(conflicts_per_turn)
```

#### 3.7.2 llm_judge.py

```python
"""LLM-as-Judge evaluation for story quality."""
from src.utils.api_client import llm_client
from config import settings

JUDGE_PROMPT = """You are an expert judge evaluating an interactive fiction game session.

Rate the following dimensions from 1-10:
{dimensions}

## Game Session
{session_text}

Respond in JSON:
{{
  "scores": {{
    "narrative_quality": <1-10>,
    "consistency": <1-10>,
    "player_agency": <1-10>,
    "creativity": <1-10>,
    "pacing": <1-10>
  }},
  "overall": <1-10>,
  "feedback": "..."
}}"""

DIMENSION_DESCRIPTIONS = {
    "narrative_quality": "Writing quality, vivid descriptions, engaging prose",
    "consistency": "Logical consistency within the story world",
    "player_agency": "How meaningfully the story responds to player choices",
    "creativity": "Originality and surprise in story developments",
    "pacing": "Appropriate rhythm of tension and release",
}

class LLMJudge:
    def evaluate_session(self, session_turns: list[dict]) -> dict:
        session_text = self._format_session(session_turns)
        dim_text = "\n".join(
            f"- {dim}: {desc}" for dim, desc in DIMENSION_DESCRIPTIONS.items()
        )
        messages = [
            {"role": "system", "content": "You are a fair and analytical judge of interactive fiction quality."},
            {"role": "user", "content": JUDGE_PROMPT.format(
                dimensions=dim_text,
                session_text=session_text,
            )},
        ]
        try:
            return llm_client.chat_json(messages, temperature=0.3)
        except Exception as e:
            return {"error": str(e)}
    
    def _format_session(self, turns: list[dict]) -> str:
        lines = []
        for i, turn in enumerate(turns):
            if "player_input" in turn:
                lines.append(f"[Turn {i+1}] Player: {turn['player_input']}")
            if "story_text" in turn:
                lines.append(f"Story: {turn['story_text']}\n")
        return "\n".join(lines)
```

---

### 3.8 app.py — Gradio 前端

```python
"""Gradio UI for StoryWeaver interactive text adventure."""
import gradio as gr
from src.engine.game_engine import GameEngine

engine: GameEngine = None

def start_new_game():
    global engine
    engine = GameEngine()
    result = engine.start_game()
    
    option_btns = [opt.text for opt in result.options]
    chat_history = [{"role": "assistant", "content": result.story_text}]
    
    return (
        chat_history,        # chatbot
        result.kg_html,      # kg_display
        str(result.nlu_debug),  # nlu_debug
        gr.update(choices=option_btns, visible=True),  # option_radio
    )

def player_action(message, chat_history, selected_option):
    global engine
    if engine is None:
        return chat_history, "", "", gr.update()
    
    # Use typed message or selected option
    player_input = message if message else selected_option
    if not player_input:
        return chat_history, "", "", gr.update()
    
    chat_history.append({"role": "user", "content": player_input})
    
    result = engine.process_turn(player_input)
    
    chat_history.append({"role": "assistant", "content": result.story_text})
    
    if result.conflicts:
        conflict_text = "\n⚠️ ".join(str(c) for c in result.conflicts)
        chat_history.append({"role": "assistant", "content": f"⚠️ Conflicts detected:\n{conflict_text}"})
    
    option_btns = [opt.text for opt in result.options]
    
    return (
        chat_history,
        result.kg_html,
        str(result.nlu_debug),
        gr.update(choices=option_btns, visible=True),
    )

def build_ui():
    with gr.Blocks(title="StoryWeaver", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏰 StoryWeaver — Interactive Text Adventure")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=500, label="Story")
                
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="Type your action...", scale=4, label="Your Action")
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                
                option_radio = gr.Radio(choices=[], label="Quick Actions", visible=False)
                start_btn = gr.Button("🎮 New Game", variant="secondary")
            
            with gr.Column(scale=2):
                kg_display = gr.HTML(label="Knowledge Graph")
                nlu_debug = gr.Textbox(label="NLU Debug", lines=8, interactive=False)
        
        # Event bindings
        start_btn.click(start_new_game, outputs=[chatbot, kg_display, nlu_debug, option_radio])
        send_btn.click(player_action, inputs=[msg_input, chatbot, option_radio], outputs=[chatbot, kg_display, nlu_debug, option_radio])
        msg_input.submit(player_action, inputs=[msg_input, chatbot, option_radio], outputs=[chatbot, kg_display, nlu_debug, option_radio])
        option_radio.change(player_action, inputs=[gr.Textbox(value="", visible=False), chatbot, option_radio], outputs=[chatbot, kg_display, nlu_debug, option_radio])
    
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=7860, share=False)
```

---

## 四、依赖清单 (requirements.txt)

```
# --- Core ---
pydantic-settings>=2.0
python-dotenv>=1.0

# --- API ---
openai>=1.0

# --- NLU ---
transformers>=4.30
torch>=2.0
spacy>=3.5
fastcoref>=2.1

# --- Knowledge Graph ---
networkx>=3.0
pyvis>=0.3

# --- UI ---
gradio>=4.0

# --- Evaluation ---
nltk>=3.8

# --- Dev ---
pytest>=7.0
```

---

## 五、四周开发进度计划

### 团队角色 (6 人)

| 角色 | 负责模块 |
|------|---------|
| A - NLU 工程师 | intent_classifier, entity_extractor, coreference, train_intent |
| B - KG 工程师 | graph, relation_extractor, conflict_detector, visualizer |
| C - NLG 工程师 | api_client, prompt_templates, story_generator, option_generator |
| D - 引擎 & UI | game_engine, state, app.py (Gradio) |
| E - 评估 & 测试 | metrics, llm_judge, all tests, evaluation pipeline |
| F - 基础设施 & 文档 | config, CI, .env, README, report, demo |

### 时间线

| 周 | 里程碑 | 交付物 |
|----|--------|--------|
| **W1** | 基础搭建 | config.py, api_client.py, graph.py 骨架, 单元测试框架, .env.example |
| **W2** | 核心模块 | NLU 三件套完成, KG CRUD + 关系抽取, NLG prompt 模板 + story_gen + option_gen |
| **W3** | 集成联调 | game_engine 串联, Gradio UI, 冲突检测, KG 可视化, 端到端测试 |
| **W4** | 评估打磨 | LLM-as-Judge, 自动评估跑分, Bug 修复, README, 最终报告, Demo 录制 |

### 每周详细任务

#### Week 1 (基础搭建)
- [ ] F: 创建 `.env.example`, 配置 `config.py` (Pydantic Settings)
- [ ] C: 实现 `api_client.py` (单例, 重试, cost tracking)
- [ ] B: 实现 `graph.py` (MultiDiGraph, add/get/remove, to_summary)
- [ ] E: 搭建 pytest 框架, 编写 api_client 和 graph 单元测试
- [ ] A: 准备意图分类训练数据 (每类 200+ 样本)
- [ ] D: `state.py` 数据类

#### Week 2 (核心模块)
- [ ] A: 微调 RoBERTa 意图分类器 + rule_fallback
- [ ] A: 实现 entity_extractor (spaCy NER + noun phrases)
- [ ] A: 实现 coreference resolver (fastcoref)
- [ ] B: 实现 relation_extractor (LLM-based)
- [ ] B: 实现 conflict_detector (规则层)
- [ ] C: 编写 prompt_templates (SYSTEM, CONTINUE, OPTION, OPENING)
- [ ] C: 实现 story_generator + option_generator
- [ ] E: NLU / KG / NLG 模块单元测试

#### Week 3 (集成联调)
- [ ] D: 实现 game_engine (NLU→KG→NLG 流水线, TurnResult)
- [ ] D: 实现 Gradio UI (chat + options + KG viz + debug)
- [ ] B: 实现 conflict_detector LLM 层
- [ ] B: 实现 visualizer (PyVis, entity styles)
- [ ] E: 端到端集成测试 (mock API)
- [ ] ALL: 联调测试, 修 Bug

#### Week 4 (评估打磨)
- [ ] E: 实现 metrics.py (distinct-n, self-BLEU, coverage, consistency)
- [ ] E: 实现 llm_judge.py (5 维度评分)
- [ ] E: 运行完整评估 (≥10 局游戏)
- [ ] F: README, 最终报告撰写
- [ ] D: UI 美化, 错误处理优化
- [ ] ALL: Demo 录制, 最终提交

---

## 六、评估标准

### 6.1 自动指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Distinct-2 | ≥ 0.75 | 二元组多样性 |
| Self-BLEU-4 | ≤ 0.35 | 低重复度 (越低越好) |
| Consistency Rate | ≥ 0.85 | 无冲突回合比例 |
| Entity Coverage | ≥ 0.60 | KG 实体在故事中的引用率 |

### 6.2 LLM-as-Judge (5 维度，1-10 分)

| 维度 | 目标 | 说明 |
|------|------|------|
| Narrative Quality | ≥ 7 | 文笔、描写、沉浸感 |
| Consistency | ≥ 7 | 故事世界逻辑一致性 |
| Player Agency | ≥ 7 | 玩家选择对故事的影响 |
| Creativity | ≥ 6 | 故事发展的创意和惊喜 |
| Pacing | ≥ 6 | 紧张与舒缓的节奏感 |

### 6.3 系统指标

| 指标 | 目标 |
|------|------|
| 平均响应时间 | ≤ 5s/turn |
| API 费用 | ≤ $0.01/game |
| Intent Accuracy | ≥ 85% |
| 端到端完整游戏 | ≥ 30 turns without crash |

---

## 七、硬件与环境要求

### 最低配置
- **GPU**: 不需要 (所有生成通过 API)
- **RAM**: 8GB (NLU 模型约 1.5GB)
- **存储**: 10GB (模型权重 + 依赖)
- **网络**: 稳定的互联网连接 (API 调用)

### 推荐配置
- **RAM**: 16GB
- **Python**: 3.10+
- **OS**: Windows/macOS/Linux

### 费用估算
- gpt-4o-mini: $0.15/1M input, $0.60/1M output
- 每回合约: ~$0.0003
- 每局 (30 回合): ~$0.009
- 项目全程 (开发+测试+评估): < $20

---

## 八、关键设计决策

1. **为什么用 MultiDiGraph 而非 DiGraph**: 同一对实体间可能存在多种关系 (如 A `ally_of` B 且 A `located_at` B 所在地)
2. **为什么 NLU 本地运行**: 课程要求展示 NLP 技术栈，且意图分类和 NER 延迟要求低
3. **为什么 NLG 使用 API**: GPT-2 本地生成质量不足以支撑良好的游戏体验，gpt-4o-mini 费用极低
4. **为什么用规则+LLM 双层冲突检测**: 规则层零延迟捕获明显矛盾，LLM 层处理复杂逻辑推理
5. **为什么 Prompt 版本化**: 方便 A/B 测试和迭代优化
6. **to_summary() 方法**: 将 KG 转为文本注入 LLM prompt，是 KG 与 NLG 的桥梁

---

*文档结束*
