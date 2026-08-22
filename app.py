import os
import re
import uuid
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gynaecology & Obstetrics Assistant",
    page_icon="🩺",
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

# --- 4. INITIALIZE VECTOR DB & LLM ---
@st.cache_resource
def init_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./dutta_vector_db",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="qwen/qwen3.6-27b",
        temperature=0.1,
        max_tokens=3000
    )
    return vector_db, llm

vector_db, llm = init_rag()

# --- 5. MODERN UI & CLINICAL CSS ---
st.markdown("""
<style>
    /* Global Reset & Modern Background */
    html, body, .stApp {
        overscroll-behavior-y: none !important;
        touch-action: pan-x pan-y !important;
        background: #f4f7fb !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Hide Unnecessary Streamlit UI Chrome */
    header[data-testid="stHeader"] { background: transparent !important; z-index: 99 !important; }
    footer, #MainMenu, [data-testid="stStatusWidget"], .stAppBadge, [data-testid="stToolbar"], 
    div[class*="viewerBadge"], button[title="View source on GitHub"], .stActionButton, 
    .stAppHostBadge, iframe[title="Streamlit App Badge"], #manage-app-button, 
    div[data-testid="stDecoration"] { display: none !important; }

    button[data-testid="stHeaderNavButton"], button[data-testid="baseButton-header"] {
        display: block !important; visibility: visible !important; opacity: 1 !important;
    }

    /* Container Spacing */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 820px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 140px !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Main Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0e7490 0%, #1e3a8a 100%);
        border-radius: 20px;
        padding: 24px 20px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(14, 116, 144, 0.25);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: clamp(1.5rem, 4vw, 2.2rem);
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 0.88rem;
        opacity: 0.9;
        margin-top: 2px;
    }

    /* Quick Action Chips */
    .chip-container {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .chip-btn {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #334155;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* Modern Chat Message Styling */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03) !important;
    }

    /* Mobile Table Responsiveness */
    div[data-testid="stChatMessage"] table {
        display: block !important;
        overflow-x: auto !important;
        white-space: normal !important;
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 12px 0 !important;
    }
    div[data-testid="stChatMessage"] th {
        min-width: 120px !important;
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        padding: 10px 12px !important;
    }
    div[data-testid="stChatMessage"] td {
        min-width: 180px !important;
        padding: 10px 12px !important;
        border-top: 1px solid #f1f5f9 !important;
    }

    /* Chat Input Fixed Bar */
    .stChatInputContainer {
        border-radius: 28px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        padding: 4px !important;
        width: 90% !important;
        max-width: 780px !important;
        position: fixed !important;
        bottom: 18px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999 !important;
    }

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px !important;
        border: none !important;
    }
    .sidebar-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
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
    if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.session_state.view_mode = "chat"
        st.rerun()

    st.markdown('<div class="sidebar-header">Recent History</div>', unsafe_allow_html=True)

    for cid, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"][:22] + "..." if len(chat_data["title"]) > 22 else chat_data["title"]
        is_active = (cid == st.session_state.current_chat_id and st.session_state.view_mode == "chat")
        btn_label = f"💬 {title}" if is_active else f"📄 {title}"
        
        if st.button(btn_label, key=f"chat_nav_{cid}", use_container_width=True):
            st.session_state.current_chat_id = cid
            st.session_state.view_mode = "chat"
            st.rerun()

    st.divider()

    with st.expander("🔒 Admin Control Panel"):
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

    # --- HERO BANNER ON EMPTY CHAT ---
    if len(messages) == 0:
        st.markdown("""
        <div class="hero-card">
            <h1 class="hero-title">Gynaecology & Obstetrics Assistant</h1>
            <div class="hero-badge">Clinical Reference Assistant</div>
            <div class="hero-sub">Developed by <b>Aryan Jadhav</b> | Knowledge Base: <b>DC Dutta</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**💡 Suggested Sample Queries for Practice:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        if col_q1.button("🚨 Ectopic Pregnancy Signs (Hinglish)", use_container_width=True):
            st.session_state["preset_query"] = "Ectopic pregnancy ke warning signs simple Hinglish me batao"
            st.rerun()
        if col_q2.button("🩺 Pre-eclampsia Management", use_container_width=True):
            st.session_state["preset_query"] = "Clinical management of severe pre-eclampsia"
            st.rerun()
        if col_q3.button("🩸 Postpartum Hemorrhage", use_container_width=True):
            st.session_state["preset_query"] = "PPH ke main causes aur immediate steps batao"
            st.rerun()

    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Handle preset quick queries or normal user chat input
    preset_query = st.session_state.pop("preset_query", None)
    chat_input_val = st.chat_input("Ask a clinical question (English or Hinglish)...")
    prompt = preset_query or chat_input_val

    if prompt:
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]

        messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Searching DC Dutta & retrieving clinical guidance..."):
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
                    page_numbers = set()
                    
                    for doc in docs:
                        meta = doc.metadata or {}
                        page_val = meta.get("page") or meta.get("page_number") or meta.get("source_page")
                        
                        if page_val is not None and str(page_val).strip() != "":
                            try:
                                p_int = int(page_val) + 1
                                page_numbers.add(str(p_int))
                                p_str = str(p_int)
                            except ValueError:
                                page_numbers.add(str(page_val))
                                p_str = str(page_val)
                        else:
                            p_str = "N/A"
                        
                        context_blocks.append(f"[DC Dutta Page {p_str}]\n{doc.page_content[:800]}")

                    context_text = "\n\n".join(context_blocks)
                    
                    if page_numbers:
                        pages_ref = ", ".join(sorted(page_numbers, key=lambda x: int(x) if x.isdigit() else 0))
                        citation_line = f"📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})"
                    else:
                        citation_line = "📌 **Source Citation**: DC Dutta Obstetrics & Gynecology Textbook"

                    full_prompt = f"""You are a senior clinical professor and medical expert explaining concepts derived from DC Dutta's Textbook of Obstetrics & Gynecology.

CONVERSATION HISTORY:
{history_context}

TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{prompt}

INSTRUCTIONS:
1. DUAL LANGUAGE ADAPTABILITY (CRITICAL):
   - Detect the language of the USER QUESTION.
   - If the user asks in Hinglish (e.g., "warning signs kya hai", "treatment batao", "kaise pehchane", "hinglish me batao"), YOU MUST RESPOND ENTIRELY IN EASY, PRACTICAL HINGLISH (Roman Hindi). Use simple terms suitable for ASHA workers and grassroots healthcare providers.
   - If the user asks in standard English, respond in clear clinical English.

2. CONTENT STRUCTURE:
   - Provide clear, structured sections (Etiology, Clinical Features/Warning Signs, Investigations, Management) as appropriate.
   - Use bullet points for readability on mobile screens.

3. CITATION:
   - Strictly append `{citation_line}` at the very end of your response.
"""
                    response = llm.invoke(full_prompt)
                    response_text = response.content

                    if "<think>" in response_text:
                        response_text = re.sub(
                            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
                        ).strip()

                except Exception as e:
                    response_text = f"⚠️ Error generating response: {str(e)}"

                st.markdown(response_text)
                messages.append({"role": "assistant", "content": response_text})
                st.rerun()
