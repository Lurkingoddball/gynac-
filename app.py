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
    # 1. Format previous chat memory BEFORE adding current prompt
    chat_history_str = ""
    recent_messages = st.session_state.messages[-6:]  # Get last 6 turns correctly
    for msg in recent_messages:
        role_label = "Student" if msg["role"] == "user" else "Tutor"
        chat_history_str += f"{role_label}: {msg['content']}\n"

    if not chat_history_str:
        chat_history_str = "None (This is the start of the conversation)."

    # Append current user prompt to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing DC Dutta & generating detailed response..."):
            # 2. Retrieve expanded textbook context (k=12 for full-chapter coverage)
            docs = vector_db.similarity_search(prompt, k=12)
            
            context_blocks = []
            page_numbers = set()
            
            for doc in docs:
                page_num = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
                if page_num != "N/A":
                    if isinstance(page_num, int):
                        page_num = page_num + 1
                    page_numbers.add(str(page_num))
                
                context_blocks.append(f"[Page {page_num}]\n{doc.page_content}")

            context_text = "\n\n".join(context_blocks)
            pages_ref = ", ".join(sorted(page_numbers)) if page_numbers else "DC Dutta Textbook"

            # 3. High-depth system prompt
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Your task is to provide comprehensive, medical-school grade explanations. Do NOT provide brief summaries. Give deep, exhaustive clinical details.

=== CONVERSATION HISTORY (FOR CONTINUITY) ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **Maintain Continuity**: Use the conversation history to understand follow-up questions, pronouns (it, this, that), or requests for clarification.
2. **Exhaustive Structure**: Organize the answer thoroughly using standard medical headings where applicable:
   - **Definition / Overview**
   - **Etiology & Risk Factors / Pathophysiology**
   - **Clinical Features & Diagnosis**
   - **Management / Line of Treatment** (Medical, Surgical, Emergency)
3. **Detail & Depth**: Use bullet points, bold key medical terms, and provide complete clinical protocols.
4. **Source Citation**: At the very end of your response, add:
   `📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})`

Generate a detailed, full-scale medical explanation:"""

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
