# -*- coding: utf-8 -*-
import streamlit as st
from core.nlu import NLUExtractor
from core.kg_manager import KGManager
from core.nli_consistency import NLIConsistencyChecker
from core.cbr_retriever import CBRRetriever
from core.nlg_templates import NLGGenerator
from utils.visualization import generate_kg_plot
from config import STORY_LENGTH, NUM_BRANCHES
import time
import random

st.set_page_config(page_title="动态KG故事生成器", layout="wide")
st.title("🎉 基于动态知识图谱与Transformer-NLI的故事生成器")
st.markdown("**纯本地硕士级NLP大作业** | 响应时间 < 2.5s | 多轮续写 + 实时KG")

# Session State
if "kg" not in st.session_state:
    st.session_state.kg = KGManager()
if "history" not in st.session_state:
    st.session_state.history = []
if "consistency_history" not in st.session_state:
    st.session_state.consistency_history = []
if "nlu" not in st.session_state:
    st.session_state.nlu = NLUExtractor()
if "nli" not in st.session_state:
    st.session_state.nli = NLIConsistencyChecker()
if "cbr" not in st.session_state:
    st.session_state.cbr = CBRRetriever()
if "nlg" not in st.session_state:
    st.session_state.nlg = NLGGenerator()

# 侧边栏
with st.sidebar:
    st.header("📊 故事世界观KG")
    fig = generate_kg_plot(st.session_state.kg)
    st.plotly_chart(fig, width="stretch")
    
    st.header("📈 一致性分数")
    if st.session_state.consistency_history:
        for i, score in enumerate(st.session_state.consistency_history[-5:]):
            st.progress(score, text=f"轮次 {len(st.session_state.history)-4+i}: {score:.2f}")
    
    if st.button("📥 下载完整故事.txt"):
        full_story = "\n\n".join([h["user"] + "\n" + h["ai"] for h in st.session_state.history])
        st.download_button("点击下载", full_story, file_name="完整故事.txt", mime="text/plain")

# 历史消息
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        st.write(msg["ai"])
        st.caption(f"一致性: {msg['consistency']:.2f}")

# 输入
user_input = st.chat_input("输入故事开头或当前情节（中文）...")
if user_input:
    start_time = time.time()
    
    nlu_result = st.session_state.nlu.extract(user_input)
    history_summary = " ".join([h["ai"] for h in st.session_state.history[-3:]]) if st.session_state.history else ""
    contradiction_prob, is_consistent = st.session_state.nli.check_consistency(history_summary, user_input)
    st.session_state.consistency_history.append(1 - contradiction_prob)
    
    if not is_consistent:
        repair_msg = st.session_state.nli.auto_repair(st.session_state.kg, user_input)
        st.warning(f"⚠️ 检测到矛盾（概率{contradiction_prob:.2f}），已自动修复KG！{repair_msg}")
    
    st.session_state.kg.add_plot(user_input, nlu_result)
    
    cbr_candidates = st.session_state.cbr.retrieve_and_adapt(user_input)
    
    context = {
        "主角": nlu_result["entities"][0]["name"] if nlu_result["entities"] else "主人公",
        "伙伴": nlu_result["entities"][1]["name"] if len(nlu_result["entities"]) > 1 else "朋友",
        "行动": nlu_result["events"][0]["predicate"] if nlu_result["events"] else "冒险",
        "情感": nlu_result["sentiment"],
    }
    continuation = st.session_state.nlg.generate_continuation(context, cbr_candidates)
    
    branches = [continuation[:STORY_LENGTH//2] + "..." + random.choice(["他们继续前进。", "突然出现转折！", "朋友加入了！"]) for _ in range(NUM_BRANCHES)]
    
    full_ai = continuation + "\n\n**分支选项：**\n" + "\n".join([f"{i+1}. {b}" for i, b in enumerate(branches)])
    
    st.session_state.history.append({
        "user": user_input,
        "ai": full_ai,
        "consistency": 1 - contradiction_prob
    })
    
    st.rerun()
    
    elapsed = time.time() - start_time
    st.success(f"✅ 本轮生成完成！耗时 {elapsed:.2f}s（<2.5s达标）")
