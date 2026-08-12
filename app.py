import os
import uuid
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gynaecology and Obstetrics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
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
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    return vector_db, llm

vector_db, llm = init_rag()

# --- 5. CUSTOM STYLING ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent !important; }
    
    .stApp {
        background: radial-gradient(circle at center, #edf4ff 0%, #ffffff 75%);
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 400;
        color: #1f1f1f;
        margin-top: 8vh;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .sub-credit {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        color: #0b57d0;
        margin-bottom: 0.2rem;
    }
    
    .source-credit {
        text-align: center;
        font-size: 0.95rem;
        color: #5f6368;
        margin-bottom: 3rem;
    }

    .stChatInputContainer {
        border-radius: 28px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e2e5 !important;
        background-color: #ffffff !important;
        padding: 4px !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    .sidebar-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .block-container {
        max-width: 850px !important;
        padding-top: 1rem !important;
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

if "admin_nav" not in st.session_state:
    st.session_state.admin_nav = "💬 Chat Interface"

# --- 7. SIDEBAR (CHAT HISTORY & ADMIN AUTH) ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.session_state.admin_nav = "💬 Chat Interface"
        st.rerun()

    st.markdown('<div class="sidebar-header">Recent Chats</div>', unsafe_allow_html=True)

    for cid, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"][:22] + "..." if len(chat_data["title"]) > 22 else chat_data["title"]
        is_active = (cid == st.session_state.current_chat_id)
        btn_label = f"💬 {title}" if not is_active else f"🗣️ {title}"
        
        if st.button(btn_label, key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid
            st.session_state.admin_nav = "💬 Chat Interface"
            st.rerun()

    st.divider()

    with st.expander("🔒 Admin Panel"):
        if not st.session_state.admin_logged_in:
            admin_user = st.text_input("Username", key="admin_user_input")
            admin_pass = st.text_input("Password", type="password", key="admin_pass_input")
            
            if st.button("Login as Admin"):
                if admin_user == ADMIN_USERNAME and admin_pass == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.success("Authenticated!")
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        else:
            st.write("🟢 **Admin Mode Active**")
            if st.button("Logout Admin"):
                st.session_state.admin_logged_in = False
                st.rerun()

# --- 8. MAIN INTERFACE & NAVIGATION ---
current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

# Navigation toggle for admin
if st.session_state.admin_logged_in:
    selected_nav = st.radio(
        "Navigation",
        ["💬 Chat Interface", "📊 Backend Analytics"],
        key="admin_nav",
        horizontal=True,
        label_visibility="collapsed"
    )
    show_analytics = (selected_nav == "📊 Backend Analytics")
else:
    show_analytics = False

# Analytics View
if st.session_state.admin_logged_in and show_analytics:
    st.title("Backend Query Logs")
    col1, col2 = st.columns(2)
    col1.metric("Active Chat Sessions", len(st.session_state.chats))
    total_q = sum(len(c["messages"]) // 2 for c in st.session_state.chats.values())
    col2.metric("Total User Queries", total_q)
    st.divider()
    for cid, data in st.session_state.chats.items():
        with st.expander(f"Session ID: {cid} - Title: {data['title']}"):
            st.json(data["messages"])

# Chat Interface View
else:
    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source: DC Dutta</div>', unsafe_allow_html=True)

    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
        messages.append({"role": "user", "content": prompt})
        # Auto-update session title from first prompt
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]
        st.rerun()

# --- 9. RAG RESPONSE GENERATION ENGINE ---
if not show_analytics and len(messages) > 0 and messages[-1]["role"] == "user":
    user_prompt = messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Searching DC Dutta & generating detailed response..."):
            try:
                # 1. Build contextual query
                recent_user_queries = [m["content"] for m in messages if m["role"] == "user"]
                search_query = f"{recent_user_queries[-2]} {user_prompt}" if len(recent_user_queries) > 1 else user_prompt

                # 2. Build history text
                chat_history_str = ""
                for msg in messages[:-1][-6:]:
                    role_label = "Student" if msg["role"] == "user" else "Tutor"
                    chat_history_str += f"{role_label}: {msg['content']}\n"
                
                if not chat_history_str:
                    chat_history_str = "None (Start of conversation)."

                # 3. Retrieve context
                docs = vector_db.similarity_search(search_query, k=10)
                
                context_blocks = []
                page_numbers = set()
                
                for doc in docs:
                    meta = doc.metadata or {}
                    page_val = None
                    for key in ["page", "page_number", "source_page", "Page", "p"]:
                        if key in meta:
                            page_val = meta[key]
                            break
                    
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
                    
                    context_blocks.append(f"[Page {p_str}]\n{doc.page_content}")

                context_text = "\n\n".join(context_blocks)
                
                if page_numbers:
                    pages_ref = ", ".join(sorted(page_numbers, key=lambda x: int(x) if x.isdigit() else 0))
                    citation_line = f"📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})"
                else:
                    citation_line = "📌 **Source Citation**: DC Dutta Obstetrics & Gynecology Textbook"

                full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Provide highly detailed, medical-school grade explanations. Do NOT provide brief summaries.

=== CONVERSATION HISTORY ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{user_prompt}

=== INSTRUCTIONS ===
1. Maintain continuity with the conversation history.
2. Structure answers with medical headings (Definition, Etiology, Clinical Features, Management).
3. Strictly include `{citation_line}` at the end.
"""
                response = llm.invoke(full_prompt)
                response_text = response.content

            except Exception as e:
                response_text = f"⚠️ Error generating response: {str(e)}"

            st.markdown(response_text)
            messages.append({"role": "assistant", "content": response_text})
