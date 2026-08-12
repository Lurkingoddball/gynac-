import streamlit as st
import uuid
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gynaecology and Obstetrics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SECURITY & ADMIN CREDENTIALS ---
ADMIN_USERNAME = "aryan_admin"
ADMIN_PASSWORD = "Aryan@2026"

# --- 3. CACHED VECTOR DATABASE LOADING ---
@st.cache_resource(show_spinner=False)
def load_vector_db():
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Ensures persistent vector loading in RAM
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="dutta_vector_db", embedding_function=embeddings)
    return db

# --- 4. CUSTOM GEMINI-STYLE CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .stApp {
        background: radial-gradient(circle at center, #f0f7ff 0%, #ffffff 70%);
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 500;
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
        margin-bottom: 0.1rem;
    }
    
    .source-credit {
        text-align: center;
        font-size: 0.9rem;
        color: #5f6368;
        margin-bottom: 3rem;
    }

    .stChatInputContainer {
        border-radius: 28px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e2e5 !important;
        background-color: #ffffff !important;
        padding: 4px !important;
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
    
    .citation-tag {
        display: inline-block;
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 5. SESSION STATE INITIALIZATION ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# --- 6. SIDEBAR MANAGEMENT ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown('<div class="sidebar-header">Recent Chats</div>', unsafe_allow_html=True)

    for cid, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"][:22] + "..." if len(chat_data["title"]) > 22 else chat_data["title"]
        is_active = (cid == st.session_state.current_chat_id)
        btn_label = f"💬 {title}" if not is_active else f"🗣️ {title}"
        
        if st.button(btn_label, key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid
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

# --- 7. MAIN INTERFACE ---
current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

if st.session_state.admin_logged_in:
    admin_tab, chat_tab = st.tabs(["📊 Admin Analytics", "💬 AI Interface View"])
else:
    admin_tab = None
    chat_tab = st.container()

if st.session_state.admin_logged_in and admin_tab:
    with admin_tab:
        st.title("Admin Dashboard & Logs")
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Sessions", len(st.session_state.chats))
        total_q = sum(len(c["messages"]) // 2 for c in st.session_state.chats.values())
        col2.metric("Total Queries", total_q)
        col3.metric("System", "Healthy 🟢")
        st.divider()
        for cid, data in st.session_state.chats.items():
            with st.expander(f"Session: {cid}"):
                st.json(data["messages"])

with (chat_tab if st.session_state.admin_logged_in else st.container()):
    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source: DC Dutta</div>', unsafe_allow_html=True)

    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "citation" in msg and msg["citation"]:
                st.markdown(f'<div class="citation-tag">📌 {msg["citation"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about Gynaecology & Obstetrics..."):
        if len(messages) == 0:
            current_chat["title"] = prompt[:25]
            
        messages.append({"role": "user", "content": prompt})
        st.rerun()

# --- 8. SEARCH & RESPONSE GENERATION ---
if len(messages) > 0 and messages[-1]["role"] == "user":
    user_prompt = messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Scanning DC Dutta textbooks..."):
            try:
                from langchain_groq import ChatGroq

                db = load_vector_db()
                
                # Retrieve top 10 relevant context chunks across both textbooks
                docs = db.similarity_search(user_prompt, k=10)
                
                context_text = "\n\n---\n\n".join([doc.page_content for doc in docs]) if docs else ""

                # Fallback if no docs retrieved
                if not context_text or len(context_text.strip()) == 0:
                    response_text = (
                        "⚠️ **No content retrieved from the database.**\n\n"
                        "Please verify that the `dutta_vector_db` folder in your GitHub repository contains the processed vector files created using `sentence-transformers/all-MiniLM-L6-v2`."
                    )
                    citation_info = "Database Notice"
                else:
                    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
                    
                    # High speed model execution
                    llm = ChatGroq(
                        temperature=0.1, 
                        groq_api_key=groq_api_key, 
                        model_name="llama-3.3-70b-versatile",
                        max_tokens=3000
                    )

                    system_prompt = f"""You are an expert clinical AI specializing exclusively in Gynaecology and Obstetrics.

STRICT RULES:
1. Rely **ONLY** on the context provided below from DC Dutta's Textbook.
2. Provide an **exhaustive, highly detailed, structured, step-by-step clinical answer**.
3. Use bold section titles, bullet points, definitions, etiology, clinical features, diagnostic steps, and management protocols from the book.
4. Do NOT add external information not present in the context.

BOOK CONTEXT:
{context_text}

QUESTION:
{user_prompt}
"""
                    response = llm.invoke(system_prompt)
                    response_text = response.content
                    citation_info = "DC Dutta Obstetrics & Gynecology Textbook"

            except Exception as e:
                response_text = f"⚠️ Error processing query: {str(e)}"
                citation_info = "System Warning"

            st.markdown(response_text)
            st.markdown(f'<div class="citation-tag">📌 Source: {citation_info}</div>', unsafe_allow_html=True)
            
            messages.append({
                "role": "assistant",
                "content": response_text,
                "citation": f"Source: {citation_info}"
            })
