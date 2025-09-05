# app_phone_agent.py
import os, re, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(layout="wide")
st.title("📱 Phone Assistant — live price, specs, reviews (Safer search)")



# --- Tools setup
# SerpAPIWrapper: restrict to India & Google domain to improve relevancy
serp = SerpAPIWrapper(
    serpapi_api_key="",
    params={"engine": "google", "gl": "IN", "google_domain": "google.co.in", "hl": "en"}
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_run = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# --- Safety helpers
BLACKLIST = {"porn", "xxx", "adult", "sex", "escort", "s-x"}  # add more as needed
def is_safe_text(txt: str) -> bool:
    t = (txt or "").lower()
    return not any(b in t for b in BLACKLIST)

def rewrite_query(q: str) -> str:
    ql = q.lower()
    # handle budget queries like "under 20000 rupees" or "below 20000"
    if ("under" in ql or "below" in ql) and ("rupee" in ql or "inr" in ql or "₹" in ql):
        # add locality, currency, year, and keywords to make search more precise
        return f"{q} best smartphones under 20000 INR India price reviews 2025"
    # general enhancement
    return q + " price reviews specs India 2025"

# --- Tool wrappers (LangChain Tool interface expects a callable that takes a string)
def serp_search_tool(query: str) -> str:
    q = rewrite_query(query)
    try:
        out = serp.run(q)  # returns text snippet(s)
    except Exception as e:
        return f"SerpAPI error: {e}"
    # very small safety filter
    if not is_safe_text(out):
        return "No safe results found."
    return out

def wiki_specs_tool(query: str) -> str:
    # expects phone model name; WikipediaQueryRun returns string
    try:
        return wiki_run.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

# convert to LangChain Tool objects
tools = [
    Tool(
        name="SerpSearch",
        func=serp_search_tool,
        description="Use for live price and reviews searches. Good for queries like 'phones under 20000 rupees' or 'iPhone 13 price India'."
    ),
    Tool(
        name="WikiSpecs",
        func=wiki_specs_tool,
        description="Use to fetch phone specifications from Wikipedia (camera, battery, display, storage)."
    ),
]

# --- LLM & agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGroq(groq_api_key="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", model_name="gemma2-9b-it", streaming=True)  # adjust model_name per your account
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True
)

# --- Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi — ask me for phone prices, specs, or reviews (India-focused)."}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Ask about phones (e.g., 'phones under 20000 rupees')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # The agent decides which tool to use (SerpSearch or WikiSpecs)
        response = agent.run(prompt, callbacks=[st_cb])
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)


