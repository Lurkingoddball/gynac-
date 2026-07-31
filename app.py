import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Set page config
st.set_page_config(
    page_title="Gynaec-Obs AI",
    page_icon="🩺",
    layout="centered"
)

# Custom Mobile Responsive CSS (Fixes off-center / right-aligned layout on phones)
st.markdown("""
<style>
    /* Full width container for mobile viewports */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2rem !important;
        max-width: 100% !important;
    }
    /* Center headings on small screens */
    @media (max-width: 768px) {
        .stAppViewContainer {
            padding: 0px !important;
        }
        h1, p {
            text-align: left;
        }
    }
</style>
""", unsafe_allow_html=True)

# Custom Header & Credits
st.title("Ask anything from Gynaec-Obs")
st.caption("Source: DC Dutta")
st.caption("By Aryan Jadhav")

# Retrieve Groq API Key
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key is missing. Please set GROQ_API_KEY in Secrets.")
    st.stop()

# Initialize Vector DB & LLM
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

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.download_button(
                label="📥 Download Note",
                data=message["content"],
                file_name=f"dutta_notes_{idx}.txt",
                mime="text/plain",
                key=f"dl_{idx}"
            )

# User input field
if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
    # 1. Capture previous context for search query
    recent_user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    
    if recent_user_queries:
        search_query = f"{recent_user_queries[-1]} {prompt}"
    else:
        search_query = prompt

    # Build conversation memory
    chat_history_str = ""
    for msg in st.session_state.messages[-6:]:
        role_label = "Student" if msg["role"] == "user" else "Tutor"
        chat_history_str += f"{role_label}: {msg['content']}\n"

    if not chat_history_str:
        chat_history_str = "None (This is the start of the conversation)."

    # Append current user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching DC Dutta & generating detailed response..."):
            # 2. Retrieve context chunks
            docs = vector_db.similarity_search(search_query, k=10)
            
            context_blocks = []
            page_numbers = set()
            
            for doc in docs:
                meta = doc.metadata or {}
                # Extract page number across all standard metadata keys
                page_num = meta.get("page", meta.get("page_number", meta.get("source_page", meta.get("Page", None))))
                
                if page_num is not None:
                    try:
                        page_int = int(page_num) + 1
                        page_numbers.add(str(page_int))
                        page_label = str(page_int)
                    except ValueError:
                        page_numbers.add(str(page_num))
                        page_label = str(page_num)
                else:
                    page_label = "Referenced Chapter"
                
                context_blocks.append(f"[Chunk Context: Page {page_label}]\n{doc.page_content}")

            context_text = "\n\n".join(context_blocks)
            
            # Format display string for source citation
            if page_numbers:
                pages_ref = "Page(s): " + ", ".join(sorted(page_numbers, key=lambda x: int(x) if x.isdigit() else 0))
            else:
                pages_ref = "DC Dutta Textbook (Standard Edition)"

            # 3. System Prompt
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Provide highly detailed, medical-school grade explanations. Do NOT give brief summaries.

=== CONVERSATION HISTORY (FOR CONTINUITY) ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **Maintain Continuity**: If the prompt is a follow-up, build directly on previous answers.
2. **Exhaustive Structure**: Use standard medical headings (Definition, Etiology & Pathophysiology, Clinical Features, Management).
3. **Source Citation Requirement**: Always end your response with an explicit source citation block:
   `📌 **Source Citation**: DC Dutta Obstetrics & Gynecology ({pages_ref})`

Generate a detailed medical response:"""

            # 4. Generate response
            response = llm.invoke(full_prompt)
            answer = response.content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            st.download_button(
                label="📥 Download Note",
                data=answer,
                file_name="dutta_notes.txt",
                mime="text/plain",
                key=f"dl_latest_{len(st.session_state.messages)}"
            )
