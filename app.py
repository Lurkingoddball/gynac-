import streamlit as st
import uuid
import datetime

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

# --- 3. CUSTOM GEMINI-STYLE CSS ---
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at center, #f0f7ff 0%, #ffffff 70%);
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Centered Greeting & Credits */
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

    /* Floating Input Styling */
    .stChatInputContainer {
        border-radius: 28px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e2e5 !important;
        background-color: #ffffff !important;
        padding: 4px !important;
    }

    /* Sidebar Customization */
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
    
    .chat-history-btn {
        width: 100%;
        text-align: left;
        border: none;
        background: transparent;
        padding: 8px 12px;
        border-radius: 8px;
        color: #3c4043;
        font-size: 0.9rem;
    }
    
    /* Citation Box */
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

# --- 4. SESSION STATE INITIALIZATION ---
if "chats" not in st.session_state:
    # Stores structure: {chat_id: {"title": str, "messages": []}}
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# --- 5. SIDEBAR: CHAT HISTORY & ADMIN ACCESS ---
with st.sidebar:
    # Top Action: New Chat Button
    if st.button("➕ New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown('<div class="sidebar-header">Recent Chats</div>', unsafe_allow_html=True)

    # Render List of Previous Chat Sessions
    for cid, chat_data in list(st.session_state.chats.items())[::-1]:
        title = chat_data["title"][:22] + "..." if len(chat_data["title"]) > 22 else chat_data["title"]
        
        # Highlight Active Chat
        is_active = (cid == st.session_state.current_chat_id)
        btn_label = f"💬 {title}" if not is_active else f"🗣️ {title}"
        
        if st.button(btn_label, key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid
            st.rerun()

    st.divider()

    # Admin Panel Expander
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
                    st.error("Invalid Admin Credentials")
        else:
            st.write("🟢 **Admin Mode Active**")
            if st.button("Logout Admin"):
                st.session_state.admin_logged_in = False
                st.rerun()

# --- 6. MAIN CONTENT AREA ---
current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

# ADMIN DASHBOARD VIEW (If Admin logged in and toggled view)
if st.session_state.admin_logged_in:
    admin_tab, chat_tab = st.tabs(["📊 Admin Backend Analytics", "💬 AI Interface View"])
else:
    admin_tab = None
    chat_tab = st.container()

# Render Admin Analytics Dashboard
if st.session_state.admin_logged_in and admin_tab:
    with admin_tab:
        st.title("Admin Dashboard & System Logs")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Active Chat Sessions", len(st.session_state.chats))
        
        total_queries = sum(len(c["messages"]) // 2 for c in st.session_state.chats.values())
        col2.metric("Total Queries Processed", total_queries)
        col3.metric("System Status", "Healthy 🟢")

        st.divider()
        st.subheader("Session Log Database")
        
        # Inspect raw backend data across all user chats
        for cid, data in st.session_state.chats.items():
            with st.expander(f"Session ID: {cid} | Title: {data['title']}"):
                st.json(data["messages"])

# Main Chat View
with (chat_tab if st.session_state.admin_logged_in else st.container()):
    
    # If starting a fresh chat, display Gemini-style centered landing UI
    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology and Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source: DC Dutta</div>', unsafe_allow_html=True)

    # Render previous conversation history for the current session
    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "citation" in msg and msg["citation"]:
                st.markdown(f'<div class="citation-tag">📌 {msg["citation"]}</div>', unsafe_allow_html=True)

    # Chat Input Pill
    if prompt := st.chat_input("Ask anything about Gynaecology & Obstetrics..."):
        
        # Update Chat Session Title based on first prompt
        if len(messages) == 0:
            current_chat["title"] = prompt[:25]
            
        # Append User Message
        messages.append({"role": "user", "content": prompt})
        
        # Rerun to render landing text out / user prompt in
        st.rerun()

# --- 7. RESPONSE GENERATION TRIGGER ---
if len(messages) > 0 and messages[-1]["role"] == "user":
    user_prompt = messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Analyzing DC Dutta knowledge base..."):
            
            # --- INSERT YOUR VECTOR SEARCH / GROQ PIPELINE HERE ---
            # Standard output placeholder:
            response_text = f"This is a structured medical response regarding **{user_prompt}** based on DC Dutta."
            citation_info = "DC Dutta Obstetrics & Gynecology Textbook"
            
            st.markdown(response_text)
            st.markdown(f'<div class="citation-tag">📌 Source: {citation_info}</div>', unsafe_allow_html=True)
            
            # Save Assistant Response
            messages.append({
                "role": "assistant",
                "content": response_text,
                "citation": f"Source: {citation_info}"
            })
