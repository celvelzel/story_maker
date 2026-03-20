"""KG visualisation via PyVis with entity-type colours and shapes.

知识图谱可视化模块：使用 PyVis 库生成交互式知识图谱可视化。

功能特点：
- 基于实体类型自动分配赛博朋克风格的霓虹配色
- 支持节点渐变 SVG 图像，提升视觉美观度
- 交互式图形界面，支持拖拽、缩放等操作
- 当 PyVis 不可用时，自动降级为纯 HTML 表格

实体类型配色方案：
- person（人物）：青色 #00f0ff
- location（地点）：绿色 #39ff14
- item（物品）：金色 #ffd700
- creature（生物）：品红 #ff00aa
- event（事件）：紫色 #7b2fff
- unknown（未知）：灰色 #5a6a8a
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

import networkx as nx

logger = logging.getLogger(__name__)

# 实体类型 → 颜色/形状映射表（赛博朋克霓虹配色）
_TYPE_STYLE = {
    "person":   {"color": "#00f0ff", "shape": "dot"},   # 青色 - 人物
    "location": {"color": "#39ff14", "shape": "dot"},   # 绿色 - 地点
    "item":     {"color": "#ffd700", "shape": "dot"},   # 金色 - 物品
    "creature": {"color": "#ff00aa", "shape": "dot"},  # 品红 - 生物
    "event":    {"color": "#7b2fff", "shape": "dot"},  # 紫色 - 事件
    "unknown":  {"color": "#5a6a8a", "shape": "dot"},  # 灰色 - 未知类型
}


def _clamp(v: int) -> int:
    """将数值限制在 [0, 255] 范围内（RGB 值合法范围）。

    参数：
        v: 输入数值

    返回：
        int: 限制后的数值
    """
    return max(0, min(255, v))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """将十六进制颜色码转换为 RGB 元组。

    参数：
        hex_color: 十六进制颜色码，如 "#00f0ff"

    返回：
        tuple[int, int, int]: RGB 值元组 (R, G, B)
    """
    c = hex_color.strip().lstrip("#")
    if len(c) != 6:
        # 无效格式，返回默认灰色
        return (90, 106, 138)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _mix(color: tuple[int, int, int], toward: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """将颜色向目标颜色混合。

    参数：
        color: 起始颜色 RGB 值
        toward: 目标颜色 RGB 值
        amount: 混合比例 [0.0, 1.0]

    返回：
        tuple[int, int, int]: 混合后的 RGB 值
    """
    return (
        _clamp(int(color[0] + (toward[0] - color[0]) * amount)),
        _clamp(int(color[1] + (toward[1] - color[1]) * amount)),
        _clamp(int(color[2] + (toward[2] - color[2]) * amount)),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """将 RGB 元组转换为十六进制颜色码。

    参数：
        rgb: RGB 值元组 (R, G, B)

    返回：
        str: 十六进制颜色码，如 "#00f0ff"
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _gradient_svg_data_uri(base_color: str) -> str:
    """构建带有径向渐变的圆形 SVG 图像并编码为 data URI。

    用于 vis-network 的 circularImage 节点，使每个实体呈现出
    从中心到边缘的颜色渐变效果。

    参数：
        base_color: 基础颜色（十六进制）

    返回：
        str: 编码为 data URI 的 SVG 字符串
    """
    # 计算渐变颜色
    base = _hex_to_rgb(base_color)
    center = _rgb_to_hex(_mix(base, (255, 255, 255), 0.35))  # 中心：偏白
    edge = _rgb_to_hex(_mix(base, (0, 0, 0), 0.30))          # 边缘：偏黑
    ring = _rgb_to_hex(_mix(base, (255, 255, 255), 0.15))     # 边框：微白

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128' viewBox='0 0 128 128'>
  <defs>
    <radialGradient id='g' cx='38%' cy='35%' r='65%'>
      <stop offset='0%' stop-color='{center}'/>
      <stop offset='58%' stop-color='{base_color}'/>
      <stop offset='100%' stop-color='{edge}'/>
    </radialGradient>
  </defs>
  <circle cx='64' cy='64' r='58' fill='url(#g)' stroke='{ring}' stroke-width='4'/>
</svg>
""".strip()

    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def render_kg_html(graph: nx.MultiDiGraph, output_path: str = "kg_vis.html") -> str:
    """生成交互式 PyVis 可视化的 HTML 字符串。

    将知识图谱渲染为可交互的 HTML 可视化图形，支持拖拽、缩放等操作。
    如果 PyVis 库不可用，自动降级为纯 HTML 表格展示。

    参数：
        graph: NetworkX MultiDiGraph 知识图谱对象
        output_path: 临时 HTML 文件保存路径

    返回：
        str: HTML 文档字符串

    降级机制：
        当 PyVis 不可用时，返回包含所有边信息的 HTML 表格
    """
    try:
        from pyvis.network import Network  # type: ignore[import-untyped]
    except ImportError:
        # PyVis 不可用，使用 HTML 表格降级方案
        return _fallback_html(graph)

    # 创建 PyVis 网络对象，配置深色主题
    net = Network(height="480px", width="100%", directed=True,
                  bgcolor="#06080f", font_color="#e0e8ff")
    # 配置物理引擎：使用 ForceAtlas2 算法进行布局
    net.set_options("""
    {"physics": {"forceAtlas2Based": {"gravitationalConstant": -50,
      "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08},
      "solver": "forceAtlas2Based", "stabilization": {"iterations": 100}},
     "edges": {"arrows": {"to": {"enabled": true}}, "smooth": {"type": "curvedCW", "roundness": 0.2}}}
    """)

    # 添加所有节点
    for node, data in graph.nodes(data=True):
        etype = data.get("entity_type", "unknown")  # 获取实体类型
        style = _TYPE_STYLE.get(etype, _TYPE_STYLE["unknown"])  # 获取对应配色
        label = data.get("name", node)  # 显示名称
        # 创建渐变 SVG 图像作为节点图标
        node_image = _gradient_svg_data_uri(style["color"])
        net.add_node(
            node,
            label=label,
            shape="circularImage",
            image=node_image,
            title=f"{label} [{etype}]",  # 悬停提示
            size=24,
        )

    # 添加所有边（关系）
    for src, tgt, data in graph.edges(data=True):
        rel = data.get("relation", "related_to")  # 关系类型
        net.add_edge(src, tgt, label=rel, title=rel, width=2, color="#00f0ff55")

    # 尝试保存并读取 HTML 文件
    try:
        net.save_graph(output_path)
        with open(output_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        # 保存失败，降级为 HTML 表格
        return _fallback_html(graph)


def _fallback_html(graph: nx.MultiDiGraph) -> str:
    """生成纯 HTML 表格作为降级可视化方案。

    当 PyVis 库不可用时，使用简单的 HTML 表格展示知识图谱的边关系。
    表格包含三列：源实体、关系类型、目标实体。

    参数：
        graph: NetworkX MultiDiGraph 知识图谱对象

    返回：
        str: 包含所有边信息的 HTML 表格字符串
    """
    rows = ""
    # 遍历所有边，构建表格行
    for src, tgt, data in graph.edges(data=True):
        rel = data.get("relation", "related_to")
        rows += f"<tr><td>{src}</td><td>{rel}</td><td>{tgt}</td></tr>"
    # 返回深色主题的 HTML 表格
    return (
        "<table border='1' style='color:#e0e8ff;background:#06080f;border-color:#00f0ff33;'>"
        "<tr><th>Source</th><th>Relation</th><th>Target</th></tr>"
        f"{rows}</table>"
    )
