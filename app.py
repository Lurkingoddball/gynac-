import os
import re
import uuid
import shutil
import fitz  # PyMuPDF
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gynaecology and Obstetrics",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ADMIN CREDENTIALS ---
ADMIN_USERNAME = "aryan_admin"
ADMIN_PASSWORD = "Aryan@2026"

# --- 3. RETRIEVE GROQ API KEY ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key is missing. Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 4. DATA DIRECTORIES & INITIALIZATION ---
BOOKS_DIR = "./books"
CHROMA_DIR = "./chroma_db"

@st.cache_resource(show_spinner=False)
def init_knowledge_base():
    """Ensures DC Dutta textbooks and pre-built Chroma DB are loaded locally or from Hugging Face."""
    os.makedirs(BOOKS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    gyn_pdf = os.path.join(BOOKS_DIR, "DC-Dutta-gynecology-6th-ed.pdf")
    obs_pdf = os.path.join(BOOKS_DIR, "Dutta-Obstetrics.pdf")
    chroma_sqlite = os.path.join(CHROMA_DIR, "chroma.sqlite3")

    # If books or vector DB not yet present
    if not (os.path.exists(gyn_pdf) and os.path.exists(obs_pdf) and os.path.exists(chroma_sqlite)):
        # 1. Check local computer storage (D:\books and D:\medai\chroma_db)
        if os.path.exists("D:/books/DC-Dutta-gynecology-6th-ed.pdf") and os.path.exists("D:/medai/chroma_db/chroma.sqlite3"):
            if not os.path.exists(gyn_pdf):
                shutil.copy("D:/books/DC-Dutta-gynecology-6th-ed.pdf", gyn_pdf)
            if not os.path.exists(obs_pdf):
                shutil.copy("D:/books/Dutta-Obstetrics.pdf", obs_pdf)
            if not os.path.exists(chroma_sqlite):
                shutil.copytree("D:/medai/chroma_db", CHROMA_DIR, dirs_exist_ok=True)
        else:
            # 2. Deployed on Streamlit Cloud: download from Hugging Face dataset
            with st.spinner("📥 Downloading DC Dutta textbooks and knowledge base from Hugging Face..."):
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="Lurkingoddball/medical-ai-data",
                    repo_type="dataset",
                    local_dir=".",
                    allow_patterns=[
                        "chroma_db/**",
                        "books/DC-Dutta-gynecology-6th-ed.pdf",
                        "books/Dutta-Obstetrics.pdf"
                    ]
                )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("medical_textbooks_v1")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return collection, embedder

collection, embedder = init_knowledge_base()

# --- 5. HIGH-RESOLUTION PDF PAGE IMAGE RENDERER ---
@st.cache_data(show_spinner=False, max_entries=120)
def render_pdf_page(raw_file: str, page_num: int, dpi: int = 140) -> bytes:
    """Renders a 1-indexed page from a DC Dutta textbook PDF as crisp PNG bytes."""
    clean_name = os.path.basename(raw_file)
    for base_dir in [BOOKS_DIR, "D:/books"]:
        pdf_path = os.path.join(base_dir, clean_name)
        if os.path.exists(pdf_path):
            try:
                doc = fitz.open(pdf_path)
                page_idx = max(0, min(page_num - 1, len(doc) - 1))
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=dpi)
                return pix.tobytes("png")
            except Exception:
                pass
    return None

# --- 6. TEXTBOOK SEARCH & RETRIEVAL ---
def retrieve_dutta_pages(query: str, top_k: int = 5):
    """Searches DC Dutta Gynecology & Obstetrics in the vector DB."""
    try:
        emb = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where={"file_path": {"$in": [
                "D:\\books\\DC-Dutta-gynecology-6th-ed.pdf",
                "D:\\books\\Dutta-Obstetrics.pdf",
                "DC-Dutta-gynecology-6th-ed.pdf",
                "Dutta-Obstetrics.pdf"
            ]}}
        )
        retrieved_docs = []
        if results and results.get("documents") and results["documents"][0]:
            for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                f_path = meta.get("file_name") or meta.get("file_path", "")
                raw_name = os.path.basename(f_path)
                page_str = str(meta.get("page_label") or meta.get("source") or "1")
                try:
                    p_num = int(page_str)
                except ValueError:
                    p_num = 1

                is_gyn = "gynecology" in raw_name.lower()
                book_title = "DC Dutta Gynecology" if is_gyn else "DC Dutta Obstetrics"

                retrieved_docs.append({
                    "raw_file": raw_name,
                    "book_title": book_title,
                    "page": p_num,
                    "text": doc_text
                })
        return retrieved_docs
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

# --- 7. GROQ LLM PIPELINE WITH AUTOMATIC FALLBACK ---
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant"
]

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=CANDIDATE_MODELS[0],
    temperature=0.1,
    max_tokens=3000
)

def invoke_llm_with_fallback(prompt_text):
    """Invokes primary model and automatically falls back if model is deprecated or unavailable."""
    try:
        resp = llm.invoke(prompt_text)
        text = resp.content
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as primary_err:
        last_err = primary_err
        for model_name in CANDIDATE_MODELS[1:]:
            try:
                curr_llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=0.1,
                    max_tokens=3000
                )
                resp = curr_llm.invoke(prompt_text)
                text = resp.content
                if "<think>" in text:
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text
            except Exception as e:
                last_err = e
                continue
        raise last_err

# --- 8. CUSTOM CSS THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

    /* Global Background & Typography */
    html, body, .stApp {
        overscroll-behavior-y: none !important;
        touch-action: pan-x pan-y !important;
        background-color: #f4f1ea !important; /* Soft warm cream background */
        color: #111111 !important;
        font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit Default UI Elements */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 99 !important;
    }
    
    footer, #MainMenu, [data-testid="stStatusWidget"], .stAppBadge, 
    [data-testid="stToolbar"], div[class*="viewerBadge"], 
    button[title="View source on GitHub"], .stActionButton, 
    .stAppHostBadge, iframe[title="Streamlit App Badge"], 
    #manage-app-button, div[data-testid="stDecoration"] {
        display: none !important;
    }

    button[data-testid="stHeaderNavButton"], 
    button[data-testid="baseButton-header"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Main Container Padding */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 2.5rem !important;
        padding-bottom: 160px !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Bold Display Header Typography */
    .bold-title {
        text-align: center;
        font-size: clamp(2rem, 6vw, 3.2rem);
        font-weight: 900;
        letter-spacing: 1px;
        color: #000000;
        text-transform: uppercase;
        margin-top: 2vh;
        margin-bottom: 1.2rem;
        line-height: 1.1;
    }
    
    .sub-credit {
        text-align: center;
        font-size: clamp(1rem, 3.5vw, 1.25rem);
        font-weight: 500;
        color: #1a1a1a;
        margin-bottom: 0.4rem;
    }
    
    .source-credit {
        text-align: center;
        font-size: clamp(0.9rem, 3vw, 1.05rem);
        font-style: italic;
        color: #333333;
        margin-bottom: 2rem;
    }

    /* Styled Chat Bubble Cards */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 20px !important;
        padding: 18px 22px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    }

    /* Custom Input Bar Styling */
    .stChatInputContainer {
        border-radius: 35px !important;
        border: 2px solid #000000 !important;
        background-color: #ffffff !important;
        padding: 6px 12px !important;
        width: 90% !important;
        max-width: 800px !important;
        position: fixed !important;
        bottom: 25px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06) !important;
    }

    .stChatInputContainer textarea {
        font-size: 1.05rem !important;
        color: #111111 !important;
    }

    /* Mobile Table Layout Fixes */
    div[data-testid="stChatMessage"] table {
        display: block !important;
        overflow-x: auto !important;
        white-space: normal !important;
        width: 100% !important;
        border-collapse: collapse !important;
        margin-top: 12px !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stChatMessage"] th {
        min-width: 120px !important;
        background-color: #e8e4dc !important;
        color: #000000 !important;
        font-weight: 700 !important;
        padding: 10px 14px !important;
        text-align: left !important;
        border: 1px solid #111111 !important;
    }

    div[data-testid="stChatMessage"] td {
        min-width: 180px !important;
        padding: 10px 14px !important;
        border: 1px solid #cbd5e1 !important;
        vertical-align: top !important;
    }

    /* Page Preview Card Styling */
    .page-preview-box {
        background-color: #faf9f6;
        border: 1.5px solid #d5d0c5;
        border-radius: 12px;
        padding: 14px;
        margin-top: 14px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #eae6dd !important;
        border-right: 2px solid #000000 !important;
        z-index: 99999 !important;
    }

    .sidebar-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #444444;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 9. SESSION STATE INITIALIZATION ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"

if st.session_state.current_chat_id not in st.session_state.chats:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

# --- 10. SIDEBAR (NAVIGATION & ADMIN) ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.session_state.view_mode = "chat"
        st.rerun()

    st.markdown('<div class="sidebar-header">Recent Chats</div>', unsafe_allow_html=True)

    for cid, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"][:20] + "..." if len(chat_data["title"]) > 20 else chat_data["title"]
        is_active = (cid == st.session_state.current_chat_id and st.session_state.view_mode == "chat")
        btn_label = f"🗣️ {title}" if is_active else f"💬 {title}"
        
        if st.button(btn_label, key=f"chat_nav_{cid}", use_container_width=True):
            st.session_state.current_chat_id = cid
            st.session_state.view_mode = "chat"
            st.rerun()

    st.divider()

    with st.expander("🔒 Admin Panel"):
        if not st.session_state.admin_logged_in:
            admin_user = st.text_input("Username", key="admin_user_input")
            admin_pass = st.text_input("Password", type="password", key="admin_pass_input")
            
            if st.button("Login as Admin", use_container_width=True):
                if admin_user == ADMIN_USERNAME and admin_pass == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.success("Authenticated!")
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        else:
            st.write("🟢 **Admin Authenticated**")
            col_a, col_b = st.columns(2)
            if col_a.button("📊 Analytics", use_container_width=True):
                st.session_state.view_mode = "analytics"
                st.rerun()
            if col_b.button("💬 Chat Mode", use_container_width=True):
                st.session_state.view_mode = "chat"
                st.rerun()
                
            if st.button("Logout Admin", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.session_state.view_mode = "chat"
                st.rerun()

# --- 11. UI HELPER: INTERACTIVE TEXTBOOK PAGE VIEWER ---
def render_page_viewer(pages_list, msg_idx: int):
    """Renders the textbook page citation badge and interactive visual page preview."""
    if not pages_list:
        return

    # Deduplicate pages based on (raw_file, page)
    unique_pages = []
    seen = set()
    for p in pages_list:
        key = (p.get("raw_file", ""), p.get("page", 0))
        if key not in seen:
            seen.add(key)
            unique_pages.append(p)

    with st.expander("📖 Original DC Dutta Textbook Page Preview", expanded=False):
        options = [f"📘 {p['book_title']} — Page {p['page']}" for p in unique_pages]
        selected_idx = st.selectbox(
            "Select textbook page to inspect original diagrams & text:",
            range(len(options)),
            format_func=lambda i: options[i],
            key=f"page_select_{msg_idx}"
        )
        selected_page = unique_pages[selected_idx]

        img_bytes = render_pdf_page(selected_page["raw_file"], selected_page["page"])
        if img_bytes:
            st.image(
                img_bytes,
                caption=f"Authentic Page from {selected_page['book_title']} (Page {selected_page['page']})",
                use_container_width=True
            )
            col1, _ = st.columns([1, 2])
            with col1:
                st.download_button(
                    label=f"💾 Download Page {selected_page['page']} (PNG)",
                    data=img_bytes,
                    file_name=f"{selected_page['book_title']}_page_{selected_page['page']}.png",
                    mime="image/png",
                    key=f"dl_btn_{msg_idx}_{selected_idx}",
                    use_container_width=True
                )
        else:
            st.info(f"Page image unavailable for {selected_page['book_title']} (Page {selected_page['page']}).")

# --- 12. MAIN VIEW ROUTING ---
if st.session_state.admin_logged_in and st.session_state.view_mode == "analytics":
    st.title("⚙️ Backend Admin Analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Sessions", len(st.session_state.chats))
    
    total_q = sum(len([m for m in c["messages"] if m["role"] == "user"]) for c in st.session_state.chats.values())
    col2.metric("Total User Queries", total_q)
    
    total_messages = sum(len(c["messages"]) for c in st.session_state.chats.values())
    col3.metric("Total Messages Exchanged", total_messages)
    
    st.divider()
    st.subheader("Session Log Transcripts")
    
    for cid, data in st.session_state.chats.items():
        with st.expander(f"Session: {data['title']} (ID: {cid[:8]}...)"):
            st.write(f"**Full Title:** {data['title']}")
            st.write(f"**Total Messages:** {len(data['messages'])}")
            st.json(data["messages"])

else:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if len(messages) == 0:
        st.markdown('<div class="bold-title">GYNAECOLOGY AND OBSTETRICS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source : DC Dutta Gynecology & Obstetrics</div>', unsafe_allow_html=True)

    # Render previous messages
    for msg_idx, msg in enumerate(messages):
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("pages"):
                render_page_viewer(msg["pages"], msg_idx)

    # Chat Input processing
    if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]

        messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Searching DC Dutta textbooks & retrieving authentic pages..."):
                try:
                    user_queries = [m["content"] for m in messages if m["role"] == "user"]
                    last_user_topic = user_queries[-2] if len(user_queries) >= 2 else ""
                    
                    search_query = f"{last_user_topic} {prompt}".strip()
                    retrieved_docs = retrieve_dutta_pages(search_query, top_k=5)

                    history_context = ""
                    for m in messages[:-1][-4:]:
                        role_str = "Student" if m["role"] == "user" else "Tutor"
                        content_str = m['content'] if m['role'] == 'user' else m['content'][:150] + "..."
                        history_context += f"{role_str}: {content_str}\n"
                    
                    if not history_context:
                        history_context = "None"

                    context_blocks = []
                    page_citations = []
                    pages_for_viewer = []

                    for doc in retrieved_docs:
                        b_title = doc["book_title"]
                        p_num = doc["page"]
                        r_file = doc["raw_file"]
                        citation_tag = f"{b_title} (Page {p_num})"
                        if citation_tag not in page_citations:
                            page_citations.append(citation_tag)
                        pages_for_viewer.append({
                            "raw_file": r_file,
                            "book_title": b_title,
                            "page": p_num
                        })
                        context_blocks.append(f"[{b_title} — Page {p_num}]\n{doc['text'][:800]}")

                    context_text = "\n\n".join(context_blocks)
                    
                    if page_citations:
                        citation_line = f"📌 **Source Citation**: {', '.join(page_citations)}"
                    else:
                        citation_line = "📌 **Source Citation**: DC Dutta Textbook of Obstetrics & Gynecology"

                    full_prompt = f"""You are a senior clinical professor and medical expert explaining concepts derived strictly from DC Dutta's Textbooks of Obstetrics & Gynecology.

CONVERSATION HISTORY:
{history_context}

DC DUTTA TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{prompt}

INSTRUCTIONS:
1. DUAL LANGUAGE ADAPTABILITY (CRITICAL):
   - Detect the language of the USER QUESTION.
   - If the user asks in Hinglish (e.g., "warning signs kya hai", "treatment batao", "kaise pehchane", "hinglish me batao"), YOU MUST RESPOND ENTIRELY IN EASY, PRACTICAL HINGLISH (Roman Hindi). Use simple terms suitable for ASHA workers and grassroots healthcare providers.
   - If the user asks in standard English, respond in clear clinical English.

2. CONTENT STRUCTURE:
   - Provide clear, structured sections (Definition/Overview, Etiology/Pathophysiology, Clinical Features/Warning Signs, Investigations, Management & Treatment Protocol) as appropriate.
   - Use bullet points and markdown bolding for high-yield clarity on mobile screens.

3. MANDATORY CITATION:
   - Strictly append this exact citation line at the very end of your response:
   {citation_line}
"""
                    response_text = invoke_llm_with_fallback(full_prompt)

                except Exception as e:
                    response_text = f"⚠️ Error generating response: {str(e)}"
                    pages_for_viewer = []

                st.markdown(response_text)
                if pages_for_viewer:
                    render_page_viewer(pages_for_viewer, len(messages))

                messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "pages": pages_for_viewer
                })
                st.rerun()
