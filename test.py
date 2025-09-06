# app_phone_agent.py
import os, re, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

# -------------------
# Streamlit UI setup
# -------------------
st.set_page_config(layout="wide")
st.title("📱 Phone Assistant — Live Price, Specs & Images")

# -------------------
# Tools setup
# -------------------

# Text search: Google search for price/reviews
serp_text = SerpAPIWrapper(
    serpapi_api_key="",  # 🔑 Put your SERPAPI key here
    params={"engine": "google", "gl": "IN", "google_domain": "google.co.in", "hl": "en"}
)

# Image search: Google Images
serp_images = SerpAPIWrapper(
    serpapi_api_key="",  # 🔑 Put your SERPAPI key here
    params={"engine": "google_images", "gl": "IN", "hl": "en"}
)

# Wikipedia for specs
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_run = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Safety filter (block unsafe results)
BLACKLIST = {"porn", "xxx", "adult", "sex", "escort"}
def is_safe_text(txt: str) -> bool:
    return not any(b in (txt or "").lower() for b in BLACKLIST)

# Helper to improve queries
def rewrite_query(q: str) -> str:
    ql = q.lower()
    if ("under" in ql or "below" in ql) and ("rupee" in ql or "inr" in ql or "₹" in ql):
        return f"{q} best smartphones under 20000 INR India price reviews 2025"
    return q + " price reviews specs India 2025"

# Tool functions
def serp_search_tool(query: str) -> str:
    q = rewrite_query(query)
    try:
        out = serp_text.run(q)
    except Exception as e:
        return f"SerpAPI error: {e}"
    return out if is_safe_text(out) else "No safe results found."

def phone_image_tool(query: str) -> str:
    try:
        result = serp_images.run(query + " smartphone India")
        return result
    except Exception as e:
        return f"Image search error: {e}"

def wiki_specs_tool(query: str) -> str:
    try:
        return wiki_run.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

# LangChain tools
tools = [
    Tool(
        name="SerpSearch",
        func=serp_search_tool,
        description="Use for live price and reviews searches. E.g. 'phones under 20000 rupees' or 'iPhone 13 price India'."
    ),
    Tool(
        name="WikiSpecs",
        func=wiki_specs_tool,
        description="Use to fetch phone specifications from Wikipedia (camera, battery, display, storage)."
    ),
    Tool(
        name="PhoneImage",
        func=phone_image_tool,
        description="Use to fetch an image of a phone. Input should be the phone model name."
    ),
]

# -------------------
# LLM & Agent setup
# -------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGroq(groq_api_key="", model_name="gemma2-9b-it", streaming=True)  # 🔑 Groq API key here

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True
)

# -------------------
# Chat UI
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi — ask me for phone prices, specs, or images (India-focused)."}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Ask about phones (e.g., 'phones under 20000 rupees')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

            # Extract and display image URLs (basic check)
            urls = re.findall(r'(https?://\S+)', response)
            for url in urls:
                if any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    st.image(url, width=250)
