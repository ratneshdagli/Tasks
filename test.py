# app_phone_agent.py
import os, re, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("📱 Phone Assistant — live price, specs, and images")

# ---------------- API Keys ----------------
# Store API keys in .env or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")

if not SERPAPI_KEY:
    st.warning("⚠️ Missing SerpAPI Key! Add it in your .env or Streamlit secrets.")
if not GROQ_API_KEY:
    st.warning("⚠️ Missing Groq API Key! Add it in your .env or Streamlit secrets.")

# ---------------- Tools Setup ----------------
# SerpAPI text search (Google)
serp_text = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_KEY,
    params={"engine": "google", "gl": "IN", "google_domain": "google.co.in", "hl": "en"}
)

def serp_search_tool(query: str) -> str:
    try:
        return serp_text.run(query + " India 2025 price reviews specs")
    except Exception as e:
        return f"SerpAPI text search error: {e}"

# Wikipedia for specifications
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_run = WikipediaQueryRun(api_wrapper=wiki_wrapper)

def wiki_specs_tool(query: str) -> str:
    try:
        return wiki_run.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

# SerpAPI image search
serp_images = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_KEY,
    params={"engine": "google_images", "gl": "IN", "hl": "en"}
)

def phone_image_tool(query: str) -> str:
    """Fetch top 3 phone images and return direct URLs"""
    try:
        results = serp_images.results(query + " smartphone India")
        images = results.get("images_results", [])
        if not images:
            return "No images found."
        urls = [img.get("original") or img.get("thumbnail") for img in images[:3]]
        urls = [u for u in urls if u]
        return "\n".join(urls)  # Agent output
    except Exception as e:
        return f"Image search error: {e}"

# Tools list
tools = [
    Tool(
        name="SerpSearch",
        func=serp_search_tool,
        description="Use for live price and reviews searches. Example: 'phones under 20000 rupees'"
    ),
    Tool(
        name="WikiSpecs",
        func=wiki_specs_tool,
        description="Use to fetch phone specifications (camera, battery, display, storage)."
    ),
    Tool(
        name="PhoneImages",
        func=phone_image_tool,
        description="Use to fetch phone images. Example: 'show me images of iPhone 13'"
    ),
]

# ---------------- LLM & Agent ----------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it", streaming=True)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True
)

# ---------------- Chat UI ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 — ask me for phone prices, specs, or images (India-focused)."}
    ]

# Display chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# Handle user input
if prompt := st.chat_input("Ask about phones (e.g., 'phones under 20000 rupees')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])

        if response:
            # Save agent response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

            # --- EXTRA: Extract and show images ---
            urls = re.findall(r'(https?://\S+)', response)
            img_urls = [u for u in urls if any(ext in u.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"])]
            if img_urls:
                st.write("📸 Images:")
                cols = st.columns(len(img_urls))
                for i, u in enumerate(img_urls):
                    try:
                        with cols[i]:
                            st.image(u, width=250)
                    except:
                        st.write(f"⚠️ Could not load image: {u}")
