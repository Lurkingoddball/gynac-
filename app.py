import os
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from pdf2image import convert_from_path

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DC Dutta Gynaec-Obs Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE INITIALIZATION ---
if "chats" not in st.session_state:
    st.session_state.chats = {
        "chat_1": {
            "title": "New Chat",
            "messages": []
        }
    }
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_1"
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"
if "interaction_logs" not in st.session_state:
    st.session_state.interaction_logs = []

def log_interaction(chat_id, query, status):
    st.session_state.interaction_logs.append({
        "chat_id": chat_id,
        "query": query,
        "status": status
    })

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
    text_models = [m for m in active_models if "whisper" not in m.lower()]
    
    if not text_models:
        st.error("No active text models found on this Groq account.")
        st.stop()

    selected_model = text_models[0]
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=selected_model,
        temperature=0.2,
        max_tokens=4000
    )
    
    return vector_db, llm

vector_db, llm = init_rag()

@st.cache_data(show_spinner=False)
def get_pdf_page_image(pdf_path, page_num):
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        return images[0] if images else None
    except Exception:
        return None

# --- 5. ENHANCED CUSTOM UI STYLING ---
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-credit {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0284C7;
        margin-bottom: 0.2rem;
    }
    .source-credit {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 2rem;
        font-style: italic;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .pdf-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 6. SIDEBAR NAVIGATION & CHAT MANAGEMENT ---
with st.sidebar:
    st.title("🎓 DC Dutta Assistant")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = f"chat_{len(st.session_state.chats) + 1}"
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.session_state.view_mode = "chat"
        st.rerun()

    st.markdown("---")
    st.caption("Recent Conversations")
    
    for c_id, c_data in list(st.session_state.chats.items()):
        btn_label = f"💬 {c_data['title']}"
        if st.button(btn_label, key=f"nav_{c_id}", use_container_width=True):
            st.session_state.current_chat_id = c_id
            st.session_state.view_mode = "chat"
            st.rerun()

    st.markdown("---")
    with st.expander("🔒 Admin Panel"):
        if not st.session_state.admin_logged_in:
            admin_pass = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if admin_pass == "admin123":
                    st.session_state.admin_logged_in = True
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Incorrect password")
        else:
            st.write("Logged in as Admin")
            if st.button("View Analytics", use_container_width=True):
                st.session_state.view_mode = "analytics"
                st.rerun()
            if st.button("Logout", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.session_state.view_mode = "chat"
                st.rerun()

# --- 7. ADMIN ANALYTICAL DASHBOARD VIEW ---
if st.session_state.admin_logged_in and st.session_state.view_mode == "analytics":
    st.title("📊 Admin Analytics Dashboard")
    st.write(f"Total Logged Interactions: {len(st.session_state.interaction_logs)}")
    st.dataframe(st.session_state.interaction_logs, use_container_width=True)
    if st.button("Back to Chat"):
        st.session_state.view_mode = "chat"
        st.rerun()

# --- 8. MAIN CHAT VIEW ---
else:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    messages = current_chat["messages"]

    if len(messages) == 0:
        st.markdown('<div class="main-title">Gynaecology & Obstetrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-credit">Made by Aryan Jadhav</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-credit">Source Citation: DC Dutta Textbook</div>', unsafe_allow_html=True)

    # Render previous messages
    for msg in messages:
        avatar = "🎓" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar):
            pdf_pages = msg.get("pdf_pages", [])
            
            if pdf_pages and msg["role"] == "assistant":
                col_text, col_pdf = st.columns([1.2, 0.8])
                
                with col_text:
                    st.markdown(msg["content"])
                    
                with col_pdf:
                    page_num = pdf_pages[0]
                    img = get_pdf_page_image("./dc_dutta.pdf", page_num)
                    if img:
                        st.image(img, caption=f"DC Dutta — Page {page_num}", use_container_width=True)
                    else:
                        st.info(f"📄 Reference Page: DC Dutta (Page {page_num})")
            else:
                st.markdown(msg["content"])

    # Chat Input processing
    if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:25]

        messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Searching DC Dutta & generating clinical breakdown..."):
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
                        # Check all standard metadata keys for page numbers
                        page_val = meta.get("page") if meta.get("page") is not None else meta.get("page_number")
                        if page_val is None:
                            page_val = meta.get("source_page")
                        
                        if page_val is not None:
                            try:
                                p_int = int(page_val) + 1
                                if p_int not in page_numbers:
                                    page_numbers.append(p_int)
                                p_str = str(p_int)
                            except ValueError:
                                p_str = str(page_val)
                        else:
                            p_str = "N/A"
                        
                        context_blocks.append(f"[DC Dutta Page {p_str}]\n{doc.page_content}")

                    context_text = "\n\n".join(context_blocks)
                    
                    if page_numbers:
                        pages_ref = ", ".join(map(str, sorted(page_numbers)))
                        citation_line = f"📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})"
                    else:
                        citation_line = "📌 **Source Citation**: DC Dutta Obstetrics & Gynecology Textbook"

                    full_prompt = f"""You are a senior Professor of Obstetrics and Gynecology providing highly detailed, comprehensive, and exhaustive exam answers derived strictly from DC Dutta's Textbook.

CONVERSATION HISTORY:
{history_context}

TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{prompt}

INSTRUCTIONS:
- Do NOT provide brief or summarized answers. Provide exhaustive, textbook-level detail covering Definition, Etiology, Pathophysiology, Clinical Features, Diagnosis, Management, and Complications where relevant.
- Use clean Markdown formatting with bold headers and clear bullet points.
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

                top_page = page_numbers[:1] if page_numbers else []

                messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "pdf_pages": top_page
                })
                st.rerun()
