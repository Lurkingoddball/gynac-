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

# --- 5. GEMINI UI STYLING ---
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
        font-size: 2.5rem;
        font-weight: 400;
        color: #1f1f1f;
        margin-top: 5vh;
        margin-bottom: 0.2rem;
    }
    
    .sub-credit {
        text-align: center;
        font-size: 1rem;
        font-weight: 500;
        color: #0b57d0;
        margin-bottom: 0.2rem;
    }
    
    .source-credit {
        text-align: center;
        font-size: 0.9rem;
        color: #5f6368;
        margin-bottom: 2rem;
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
        font-size: 0.8rem;
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

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"

# Fix session state fallback if current chat is deleted or corrupted
if st.session_state.current_chat_id not in st.session_state.chats:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

# --- 7. SIDEBAR (CHAT MANAGEMENT & ADMIN) ---
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
    # --- ADMIN ANALYTICS DASHBOARD ---
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
    # --- STUDENT CHAT INTERFACE ---
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source: DC Dutta</div>', unsafe_allow_html=True)

    # Render history
    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # User Input Handling
    if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
        # Set chat title on first message
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]

        # Append User Message
        messages.append({"role": "user", "content": prompt})
        
        # Display User Message immediately
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        # Generate Assistant Response inline
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Searching DC Dutta & generating detailed response..."):
                try:
                    # Context assembly
                    history_context = ""
                    for m in messages[:-1][-6:]:
                        role_str = "Student" if m["role"] == "user" else "Tutor"
                        history_context += f"{role_str}: {m['content']}\n"
                    
                    if not history_context:
                        history_context = "No prior context."

                    # Vector DB Query
                    docs = vector_db.similarity_search(prompt, k=10)
                    
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
{history_context}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

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
                st.rerun()
