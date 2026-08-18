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

# --- 2. ADMIN CREDENTIALS & LOG FILE ---
ADMIN_USERNAME = "aryan_admin"
ADMIN_PASSWORD = "Aryan@2026"
LOG_FILE = "global_seminar_logs.csv"

# Ensure CSV log exists
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["timestamp", "session_id", "user_query", "response_status"])
    df.to_csv(LOG_FILE, index=False)

def log_interaction(session_id, user_query, status):
    """Logs student activity to a shared global file."""
    new_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "user_query": user_query,
        "response_status": status
    }])
    new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

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
    # Using Llama 3.3 70B for higher seminar throughput
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=3000
    )
    return vector_db, llm

vector_db, llm = init_rag()

# --- 5. STYLING ---
st.markdown("""
<style>
    html, body, .stApp { overscroll-behavior-y: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; z-index: 99 !important; }
    footer, #MainMenu, [data-testid="stStatusWidget"] { display: none !important; }
    .stApp { background: radial-gradient(circle at center, #edf4ff 0%, #ffffff 75%); }
    .main .block-container { max-width: 800px !important; margin: 0 auto; padding-bottom: 140px !important; }
    .stChatInputContainer {
        border-radius: 28px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        position: fixed !important; bottom: 15px !important; left: 50% !important; transform: translateX(-50%) !important; z-index: 999 !important;
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

# --- 7. SIDEBAR (NAVIGATION & ADMIN) ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
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
    st.title("📊 Live Seminar Admin Dashboard")
    
    # Load global logs
    if os.path.exists(LOG_FILE):
        logs_df = pd.read_csv(LOG_FILE)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Seminar Devices", logs_df["session_id"].nunique())
        c2.metric("Total Questions Asked", len(logs_df))
        c3.metric("Successful Responses", len(logs_df[logs_df["response_status"] == "Success"]))
        
        st.divider()
        st.subheader("📥 Live Activity Feed")
        st.dataframe(logs_df.sort_index(ascending=False), use_container_width=True)
        
        # Download button for post-seminar review
        csv_data = logs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Seminar Log CSV",
            data=csv_data,
            file_name="seminar_chat_logs.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No seminar logs recorded yet.")

else:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if len(messages) == 0:
        st.markdown('<div style="text-align:center; font-size: 2rem; font-weight: bold;">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; color: #0b57d0;">Made by Aryan Jadhav | DC Dutta Source</div>', unsafe_allow_html=True)

    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

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

                    context_blocks = [f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content[:800]}" for doc in docs]
                    context_text = "\n\n".join(context_blocks)

                    full_prompt = f"""You are a senior Professor of Obstetrics and Gynecology providing medical exam answers derived from DC Dutta's Textbook.

CONVERSATION HISTORY:
{history_context}

TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{prompt}

INSTRUCTIONS:
- Answer the user's question directly in relation to the ongoing clinical topic.
- Provide a detailed, structured medical answer.
- Understand Hinglish/Hindi queries (e.g. "investigation batao"). Respond in clear English.
- Append source citation at the bottom.
"""
                    response = llm.invoke(full_prompt)
                    response_text = response.content

                    if "<think>" in response_text:
                        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

                    # Log success to shared admin backend
                    log_interaction(st.session_state.current_chat_id, prompt, "Success")

                except Exception as e:
                    response_text = f"⚠️ Error generating response: {str(e)}"
                    log_interaction(st.session_state.current_chat_id, prompt, f"Error: {str(e)[:50]}")

                st.markdown(response_text)
                messages.append({"role": "assistant", "content": response_text})
                st.rerun()
