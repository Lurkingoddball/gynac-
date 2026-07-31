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
    # 1. Capture recent context for search & LLM memory
    recent_user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    
    # Build a context-aware search query so follow-up queries like "how to manage it?" include the main topic
    if recent_user_queries:
        search_query = f"{recent_user_queries[-1]} {prompt}"
    else:
        search_query = prompt

    # Format previous conversation history for the LLM
    chat_history_str = ""
    for msg in st.session_state.messages[-6:]:
        role_label = "Student" if msg["role"] == "user" else "Tutor"
        chat_history_str += f"{role_label}: {msg['content']}\n"

    if not chat_history_str:
        chat_history_str = "None (This is the start of the conversation)."

    # Append current user prompt to chat display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching DC Dutta & generating detailed response..."):
            # 2. Retrieve textbook context using the combined search query
            docs = vector_db.similarity_search(search_query, k=10)
            
            context_blocks = []
            page_numbers = set()
            
            for doc in docs:
                meta = doc.metadata or {}
                # Scan all common metadata keys for page numbers
                page_num = meta.get("page", meta.get("page_number", meta.get("source_page", meta.get("Page", "N/A"))))
                
                if page_num != "N/A":
                    try:
                        # Convert 0-indexed PDF page to 1-indexed standard page
                        page_int = int(page_num) + 1
                        page_numbers.add(str(page_int))
                        page_label = str(page_int)
                    except ValueError:
                        page_numbers.add(str(page_num))
                        page_label = str(page_num)
                else:
                    page_label = "N/A"
                
                context_blocks.append(f"[Page {page_label}]\n{doc.page_content}")

            context_text = "\n\n".join(context_blocks)
            pages_ref = ", ".join(sorted(page_numbers, key=lambda x: int(x) if x.isdigit() else 0)) if page_numbers else "DC Dutta Textbook"

            # 3. Comprehensive System Prompt
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Provide highly detailed, medical-school grade explanations. Do NOT provide brief summaries.

=== CONVERSATION HISTORY (FOR CONTINUITY & FOLLOW-UPS) ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **Maintain Continuity**: If the current question is a follow-up (e.g. asking "why?", "how to manage it?", "what are the clinical features?"), refer to the topic discussed in the Conversation History.
2. **Exhaustive Structure**: Format answers with standard medical headings:
   - **Definition / Overview**
   - **Etiology & Pathophysiology**
   - **Clinical Features & Diagnosis**
   - **Management / Line of Treatment** (Medical, Surgical, Emergency)
3. **Detail & Depth**: Use bullet points, bold key terms, and provide complete clinical steps.
4. **Source Citation**: At the very end of your response, strictly add:
   `📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})`

Generate a detailed medical explanation:"""

            # 4. Generate response from Groq
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
