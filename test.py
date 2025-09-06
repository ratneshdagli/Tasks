# app_phone_agent_debug.py
import os, re, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
import requests
import json

st.set_page_config(layout="wide")
st.title("📱 Phone Assistant — images debug & display")

# ------------------- Keys -------------------
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

if not SERPAPI_KEY:
    st.warning("Missing SERPAPI_KEY (set env var or Streamlit secrets).")
if not GROQ_KEY:
    st.warning("Missing GROQ_KEY (set env var or Streamlit secrets).")

# ------------------- SerpAPI wrappers -------------------
serp_images = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_KEY,
    params={"engine": "google_images", "gl": "IN", "hl": "en"}
)

serp_text = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_KEY,
    params={"engine": "google", "gl": "IN", "google_domain": "google.co.in", "hl": "en"}
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800)
wiki_run = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# ------------------- helpers -------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

def clean_url(u: str) -> str:
    if not u:
        return None
    u = u.strip().strip("[](),")
    # sometimes SerpAPI returns data URIs or javascript:— filter
    if u.startswith("data:") or u.startswith("javascript:"):
        return None
    return u

def get_image_urls_from_serp(query: str, max_images: int = 5):
    """Return list of image URLs by parsing serp_images.results JSON.
       Also return the raw JSON for debugging."""
    try:
        resp = serp_images.results(query)
    except Exception as e:
        return {"error": str(e), "images": [], "raw": {}}
    raw = resp or {}
    images_list = []
    # images_results is the common field returned by SerpAPI google_images
    candidates = raw.get("images_results") or raw.get("image_results") or []
    # fallback: sometimes top_results or inline images are stored differently
    if not candidates and isinstance(raw, dict):
        # try to find keys that look like image lists
        for k, v in raw.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                # heuristic: objects with 'thumbnail' or 'original' keys
                if any("thumbnail" in x or "original" in x or "link" in x or "source" in x for x in v[:3]):
                    candidates = v
                    break

    for item in candidates[:max_images]:
        # common keys
        url = item.get("original") or item.get("thumbnail") or item.get("source") or item.get("link") or item.get("image")
        # some results embed nested structures
        if not url:
            # inspect nested items
            for k in ("thumbnail", "original", "image", "source", "link", "displayed_link"):
                val = item.get(k)
                if isinstance(val, str):
                    url = val
                    break
        cleaned = clean_url(url)
        if cleaned:
            images_list.append(cleaned)
    # Last-resort: crawl text snippet for urls
    if not images_list:
        text_blob = json.dumps(raw)
        urls = re.findall(r"https?://\S+\.(?:jpg|jpeg|png|webp|gif)", text_blob, flags=re.IGNORECASE)
        images_list = [clean_url(u) for u in urls]
    # dedupe
    images_list = list(dict.fromkeys([u for u in images_list if u]))
    return {"error": None, "images": images_list, "raw": raw}

def test_image_url_ok(url: str, timeout=6):
    """Quick HEAD/GET to check content-type (some hosts block HEAD)."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        ct = r.headers.get("Content-Type", "")
        ok = r.status_code == 200 and ("image" in ct or any(ext in url.lower() for ext in IMAGE_EXTS))
        r.close()
        return ok, ct, r.status_code
    except Exception as e:
        return False, str(e), None

# ------------------- LangChain tools & agent -------------------
def serp_search_tool(query: str) -> str:
    try:
        return serp_text.run(query + " India 2025 price reviews specs")
    except Exception as e:
        return f"SerpAPI text error: {e}"

def wiki_specs_tool(query: str) -> str:
    try:
        return wiki_run.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

def phone_image_tool(query: str) -> str:
    """Return newline-separated image URLs (agent-facing)."""
    data = get_image_urls_from_serp(query + " smartphone india", max_images=5)
    if data["error"]:
        return f"Image search failed: {data['error']}"
    if not data["images"]:
        return "No images found."
    return "\n".join(data["images"])

tools = [
    Tool(name="SerpSearch", func=serp_search_tool,
         description="Find price, reviews, and seller info. Use for queries like 'phones under 20000 rupees'"),
    Tool(name="WikiSpecs", func=wiki_specs_tool,
         description="Get phone specs from Wikipedia"),
    Tool(name="PhoneImages", func=phone_image_tool,
         description="Return direct image URLs for a phone model"),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="gemma2-9b-it", streaming=True)
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, handle_parsing_errors=True)

# ------------------- UI state -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi — ask me for phone prices, specs, or images (India-focused)."}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# Optional: let user test image search directly (debug)
with st.sidebar.expander("Debug / Test image search"):
    sample_query = st.text_input("Image test query", value="Redmi Note 14")
    if st.button("Run image search now"):
        res = get_image_urls_from_serp(sample_query + " smartphone india", max_images=5)
        st.write("Images found:", res["images"])
        st.write("Raw JSON (first 3 keys):")
        if isinstance(res["raw"], dict):
            keys = list(res["raw"].keys())[:10]
            for k in keys:
                st.write(k, ":", res["raw"].get(k))

# ------------------- Conversation -------------------
if prompt := st.chat_input("Ask about phones (e.g., 'phones under 20000 rupees' or 'show images of Redmi Note 14')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])

        # Save and show agent response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        # --- Attempt to extract image URLs the agent might have returned in text ---
        urls_from_response = re.findall(r"(https?://\S+)", response)
        image_urls = [u for u in urls_from_response if any(u.lower().endswith(ext) for ext in IMAGE_EXTS)]

        # If none in agent text, try to detect phone model names (simple heuristic)
        # and fetch images programmatically for those models:
        if not image_urls:
            # simple known models list - expand as required
            known_models = ["Redmi Note 14", "Motorola Edge 50 Neo", "Realme Narzo 50 Pro", "Poco X6 Pro", "Samsung Galaxy A14 5G", "iPhone 13", "OnePlus Nord"]
            found_models = []
            low = response.lower()
            for m in known_models:
                if m.lower() in low:
                    found_models.append(m)
            # also try to capture "redmi" etc with regex phrases like "redmi note 14"
            if not found_models:
                m = re.search(r"(redmi\s+note\s*\d{1,2}\w*)", low)
                if m:
                    found_models.append(m.group(1))
            # For each found model, call get_image_urls_from_serp and display
            for model in found_models:
                st.write(f"🔎 Fetching images for **{model}**...")
                data = get_image_urls_from_serp(model + " smartphone India", max_images=4)
                # debug raw JSON in expander
                with st.expander(f"Raw image JSON for {model} (debug)"):
                    st.json(data["raw"])
                # show and verify images
                if data["images"]:
                    cols = st.columns(min(len(data["images"]), 4))
                    for i, url in enumerate(data["images"]):
                        ok, ct, code = test_image_url_ok(url)
                        if ok:
                            with cols[i]:
                                st.image(url, width=240)
                                st.write(url)
                        else:
                            with cols[i]:
                                st.write("⚠️ image not reachable")
                                st.write(url)
                else:
                    st.write("No images found for", model)

        else:
            # Agent returned image URLs directly — display them
            st.write("📸 Images returned by agent:")
            cols = st.columns(min(len(image_urls), 4))
            for i, u in enumerate(image_urls):
                ok, ct, code = test_image_url_ok(u)
                if ok:
                    with cols[i]:
                        st.image(u, width=240)
                        st.write(u)
                else:
                    with cols[i]:
                        st.write("⚠️ image not reachable")
                        st.write(u)
