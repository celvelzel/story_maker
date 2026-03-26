"""Test Streamlit rerun behavior"""
import streamlit as st

st.title("Test Rerun Behavior")

# Initialize session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0
    st.session_state.kg_html = ""
    st.session_state.engine = None

st.write(f"Counter: {st.session_state.counter}")
st.write(f"Engine: {st.session_state.engine}")
st.write(f"kg_html length: {len(st.session_state.kg_html)}")

# Display KG
if st.session_state.kg_html or st.session_state.engine:
    st.write("Condition 1: TRUE")
    if st.session_state.kg_html:
        st.write("-> Rendering cached kg_html")
        st.components.v1.html(st.session_state.kg_html, height=200)
    elif st.session_state.engine:
        st.write("-> Rendering kg_html from engine")
        kg_html = "<html><body><h1>KG from Engine</h1></body></html>"
        st.session_state.kg_html = kg_html
        st.components.v1.html(kg_html, height=200)
else:
    st.write("Condition 1: FALSE")
    st.info("The knowledge graph will appear after starting a game.")

# Test button
if st.button("Test Rerun"):
    st.session_state.counter += 1
    st.session_state.engine = "mock_engine"
    st.session_state.kg_html = f"<html><body><h1>Counter: {st.session_state.counter}</h1></body></html>"
    st.write("Button clicked, calling st.rerun()...")
    st.rerun()

if st.button("Test Clear"):
    st.session_state.counter = 0
    st.session_state.engine = None
    st.session_state.kg_html = ""
    st.rerun()
