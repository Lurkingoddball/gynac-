import os
import sqlite3
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM UI BRANDING (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gynaec-Obs AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern, Clean Clinical UI
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #FAFAFC;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    
    /* Branding Header Card */
    .brand-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .brand-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .brand-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    /* Chat Container Alignment & Clean Styling */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    
    /* Hide Default Streamlit Menu artifacts for cleaner app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOCAL SQLITE DATABASE FOR PERSISTENT CHAT HISTORY
# -----------------------------------------------------------------------------
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_message(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def load_chat_history(session_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM history WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

init_db()

# -----------------------------------------------------------------------------
# 3. INITIALIZE MODELS & VECTOR STORE
# -----------------------------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key is missing. Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

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

# -----------------------------------------------------------------------------
# 4. BRANDING HEADER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="brand-header">
        <div class="brand-title">Ask anything from Gynaec-Obs</div>
        <div class="brand-subtitle">Source: DC Dutta Textbook of Obstetrics & Gynecology</div>
        <div class="brand-badge">Created by Aryan Jadhav</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR: ADMIN DASHBOARD & CHAT CONTROLS
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Session Management
    st.subheader("💬 Chat Sessions")
    if st.button("➕ New Chat Session", use_container_width=True):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    # Admin Access Control Gate
    st.subheader("🔒 Admin Access")
    admin_passkey = st.secrets.get("ADMIN_PASSKEY", "aryan123")  # Default passkey
    input_passkey = st.text_input("Enter Admin Passkey", type="password")
    
    is_admin = input_passkey == admin_passkey
    
    if is_admin:
        st.success("Admin Mode Active")
        st.markdown("**Admin Controls:**")
        if st.button("📊 View Total Saved Queries"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM history")
            count = c.fetchone()[0]
            conn.close()
            st.info(f"Total database messages: {count}")
            
        if st.button("🗑️ Clear Local Database"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.session_state.messages = []
            st.success("Database cleared!")
            st.rerun()
    else:
        if input_passkey:
            st.error("Incorrect Passkey")

# -----------------------------------------------------------------------------
# 6. CHAT SESSION MEMORY LOAD
# -----------------------------------------------------------------------------
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = load_chat_history(st.session_state.session_id)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# 7. PROMPT HANDLING & RESPONSE GENERATION
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
    # 1. Save & Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_message(st.session_state.session_id, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build Context-Aware Search Query for Follow-ups
    recent_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    search_query = f"{recent_queries[-2]} {prompt}" if len(recent_queries) > 1 else prompt

    # 3. Format Conversation History for LLM Memory
    chat_history_str = ""
    for msg in st.session_state.messages[-6:]:
        label = "Student" if msg["role"] == "user" else "Tutor"
        chat_history_str += f"{label}: {msg['content']}\n"

    with st.chat_message("assistant"):
        with st.spinner("Searching DC Dutta & synthesizing response..."):
            # Retrieve Vector DB Chunks
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

            # 4. System Prompt
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Provide highly detailed, medical-school grade explanations with explicit clinical steps.

=== CONVERSATION HISTORY (FOR CONTINUITY & FOLLOW-UPS) ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **Maintain Continuity**: Handle follow-up questions seamlessly using the conversation history context.
2. **Exhaustive Structure**: Use standard medical headings:
   - **Definition / Overview**
   - **Etiology & Pathophysiology**
   - **Clinical Features & Diagnosis**
   - **Management / Line of Treatment** (Medical, Surgical, Emergency)
3. **Detail & Depth**: Use bullet points, bold key medical terms, and state precise dosage protocols where applicable.
4. **Source Citation**: End your answer strictly with:
   `{citation_line}`

Generate a thorough medical explanation:"""

            response = llm.invoke(full_prompt)
            answer = response.content

            st.markdown(answer)
            
            # Save Assistant Response
            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_chat_message(st.session_state.session_id, "assistant", answer)
