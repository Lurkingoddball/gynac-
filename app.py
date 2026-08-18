import os
import re
import uuid
import pandas as pd
from datetime import datetime
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gynaecology and Obstetrics",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. ADMIN CREDENTIALS & GLOBAL LOG FILE ---
ADMIN_USERNAME = "aryan_admin"
ADMIN_PASSWORD = "Aryan@2026"
LOG_FILE = "global_seminar_logs.csv"

# Ensure global CSV log file exists
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["timestamp", "session_id", "user_query", "response_status"])
    df.to_csv(LOG_FILE, index=False)

def log_interaction(session_id, user_query, status):
    """Logs student activity across all devices to a shared global file."""
    try:
        new_entry = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user_query": user_query,
            "response_status": status
        }])
        new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    except Exception:
        pass

# --- 3. RETRIEVE GROQ API KEY & CLIENT ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key is missing. Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 4. DYNAMIC MODEL DISCOVERY & INITIALIZATION ---
@st.cache_resource
def init_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./dutta_vector_db",
        embedding_function=embeddings
    )
    
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    active_models = [m.id for m in client.models.list().data if getattr(m, 'active', True)]
    
    # Filter strictly for reliable text-chat model prefixes
    allowed_prefixes = ("llama", "mixtral", "gemma")
    chat_models = [
        m for m in active_models 
        if any(m.lower().startswith(p) for p in allowed_prefixes)
    ]
    
    if not chat_models:
        st.error("No active Llama/Mixtral/Gemma models found on this Groq account.")
        st.stop()

    selected_model = chat_models[0]
    
   # Replace old model string with an active supported Groq model string
llm = ChatGroq(
    model="gemma2-9b-it",  # or "llama-3.3-70b-versatile"
    groq_api_key=st.secrets["GROQ_API_KEY"],
)
    
    return vector_db, llm

vector_db, llm = init_rag()
# =========================================================
from pdf2image import convert_from_path

@st.cache_data(show_spinner=False)
def get_pdf_page_image(pdf_path, page_num):
    """Converts a specific PDF page to an image for inline display."""
    try:
        # Convert 1-based page number to 0-based index
        images = convert_from_path(
            pdf_path, 
            first_page=page_num, 
            last_page=page_num
        )
        if images:
            return images[0]
    except Exception as e:
        st.warning(f"Unable to render PDF page {page_num}: {e}")
    return None
# =========================================================
# --- 5. COMPREHENSIVE CSS & UI STYLING ---
st.markdown("""
<style>
    html, body, .stApp {
        overscroll-behavior-y: none !important;
        touch-action: pan-x pan-y !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 99 !important;
    }
    
    button[data-testid="stHeaderNavButton"], 
    button[data-testid="baseButton-header"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    footer { display: none !important; }
    #MainMenu { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .stAppBadge { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    div[class*="viewerBadge"] { display: none !important; }
    button[title="View source on GitHub"] { display: none !important; }
    .stActionButton { display: none !important; }
    .stAppHostBadge { display: none !important; }
    iframe[title="Streamlit App Badge"] { display: none !important; }
    #manage-app-button { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }

    .stApp {
        background: radial-gradient(circle at center, #edf4ff 0%, #ffffff 75%);
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 800px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 1rem !important;
        padding-bottom: 140px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    div[data-testid="stChatMessage"] {
        max-width: 100% !important;
        margin: 0 auto !important;
        word-break: break-word !important;
    }

    .main-title {
        text-align: center;
        font-size: clamp(1.6rem, 5vw, 2.5rem);
        font-weight: 500;
        color: #1f1f1f;
        margin-top: 1vh;
        margin-bottom: 0.2rem;
    }
    
    .sub-credit {
        text-align: center;
        font-size: clamp(0.85rem, 3vw, 1rem);
        font-weight: 500;
        color: #0b57d0;
        margin-bottom: 0.2rem;
    }
    
    .source-credit {
        text-align: center;
        font-size: clamp(0.75rem, 2.5vw, 0.9rem);
        color: #5f6368;
        margin-bottom: 1.5rem;
    }

    .stChatInputContainer {
        border-radius: 28px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e2e5 !important;
        background-color: #ffffff !important;
        padding: 4px !important;
        width: 92% !important;
        max-width: 760px !important;
        position: fixed !important;
        bottom: 15px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
        z-index: 99999 !important;
    }

    .sidebar-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 6. SESSION STATE INITIALIZATION ---
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

# --- 7. SIDEBAR (NAVIGATION & ADMIN) ---
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

# --- 8. MAIN VIEW ROUTING ---
if st.session_state.admin_logged_in and st.session_state.view_mode == "analytics":
    # (Keep Admin Panel Analytics logic as is)
    pass
else:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source: DC Dutta</div>', unsafe_allow_html=True)

    # Render previous messages
    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            # Check if this message has associated PDF pages to render
            pdf_pages = msg.get("pdf_pages", [])
            
            if pdf_pages:
                col_text, col_pdf = st.columns([1.1, 0.9])
                with col_text:
                    st.markdown(msg["content"])
                with col_pdf:
                    st.markdown("### 📖 Textbook Page Preview")
                    for p_num in pdf_pages:
                        img = get_pdf_page_image("./dc_dutta.pdf", p_num)
                        if img:
                            st.image(img, caption=f"DC Dutta — Page {p_num}", use_container_width=True)
            else:
                st.markdown(msg["content"])

    # Chat input processing
    if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]

        messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Searching DC Dutta & retrieving details..."):
                try:
                    user_queries = [m["content"] for m in messages if m["role"] == "user"]
                    last_user_topic = user_queries[-2] if len(user_queries) >= 2 else ""
                    search_query = f"{last_user_topic} {prompt}".strip()
                    
                    docs = vector_db.similarity_search(search_query, k=5)
                    
                    history_context = ""
                    for m in messages[:-1][-4:]:
                        role_str = "Student" if m["role"] == "user" else "Tutor"
                        content_str = m['content'] if m['role'] == 'user' else m['content'][:150] + "..."
                        history_context += f"{role_str}: {content_str}\n"
                    
                    if not history_context:
                        history_context = "None"

                    context_blocks = []
                    page_numbers = []
                    
                    for doc in docs:
                        meta = doc.metadata or {}
                        page_val = meta.get("page") or meta.get("page_number") or meta.get("source_page")
                        
                        if page_val is not None and str(page_val).strip() != "":
                            try:
                                p_int = int(page_val) + 1
                                if p_int not in page_numbers:
                                    page_numbers.append(p_int)
                                p_str = str(p_int)
                            except ValueError:
                                p_str = str(page_val)
                        else:
                            p_str = "N/A"
                        
                        context_blocks.append(f"[DC Dutta Page {p_str}]\n{doc.page_content[:800]}")

                    context_text = "\n\n".join(context_blocks)
                    
                    if page_numbers:
                        pages_ref = ", ".join(map(str, sorted(page_numbers)))
                        citation_line = f"📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})"
                    else:
                        citation_line = "📌 **Source Citation**: DC Dutta Obstetrics & Gynecology Textbook"

                    full_prompt = f"""You are a senior Professor of Obstetrics and Gynecology providing medical exam answers derived from DC Dutta's Textbook.

CONVERSATION HISTORY:
{history_context}

TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{prompt}

INSTRUCTIONS:
- Answer the user's question directly in relation to the ongoing clinical topic in the conversation history.
- Provide a detailed, structured medical answer (Etiology, Clinical Features, Diagnosis/Investigations, Management) as appropriate.
- Understand Hinglish/Hindi queries (e.g., "investigation batao"). Respond in clear, professional English.
- Strictly append `{citation_line}` at the end.
"""
                    response = llm.invoke(full_prompt)
                    response_text = response.content

                    if "<think>" in response_text:
                        response_text = re.sub(
                            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
                        ).strip()

                    log_interaction(st.session_state.current_chat_id, prompt, "Success")

                except Exception as e:
                    response_text = f"⚠️ Error generating response: {str(e)}"
                    page_numbers = []
                    log_interaction(st.session_state.current_chat_id, prompt, f"Error: {str(e)[:50]}")

                # Save response along with referenced page numbers
                top_page = page_numbers[:1] if page_numbers else [] # Renders top 1 relevant page image
                
                messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "pdf_pages": top_page
                })
                st.rerun()
