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

# Custom Header & Caption Styling
st.title("Ask anything from Gynaec-Obs")
st.caption("Source: DC Dutta")

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
        temperature=0.3
    )
    return vector_db, llm

vector_db, llm = init_rag()

# Chat memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field with updated placeholder
if prompt := st.chat_input("Ask anything from Gynaec-Obs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching DC Dutta & generating response..."):
            # 1. Retrieve top context documents with page metadata
            docs = vector_db.similarity_search(prompt, k=6)
            
            # Format retrieved text along with page numbers from document metadata
            context_blocks = []
            page_numbers = set()
            
            for doc in docs:
                # Extract page number if present in metadata (defaults to 'Unknown')
                page_num = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
                if page_num != "N/A":
                    # Convert 0-indexed page numbers to standard 1-indexed if stored as integer
                    if isinstance(page_num, int):
                        page_num = page_num + 1
                    page_numbers.add(str(page_num))
                
                context_blocks.append(f"[Source Page: {page_num}]\n{doc.page_content}")

            context_text = "\n\n".join(context_blocks)
            pages_ref = ", ".join(sorted(page_numbers)) if page_numbers else "DC Dutta Textbook"

            # 2. Format recent conversation history (last 6 turns)
            chat_history_str = ""
            recent_messages = st.session_state.messages[-6:-1]
            for msg in recent_messages:
                role_label = "Student" if msg["role"] == "user" else "Tutor"
                chat_history_str += f"{role_label}: {msg['content']}\n"

            if not chat_history_str:
                chat_history_str = "None (This is the start of the conversation)."

            # 3. System Prompt requiring page references
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based on DC Dutta's Textbook.
Provide structured, highly detailed, medical-school grade explanations. Always cite the exact page numbers from where the information was retrieved.

=== RECENT CONVERSATION HISTORY ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **In-Depth Explanation**: Provide thorough, structured answers (Definition, Etiology, Clinical Features, Management where applicable).
2. **Include Page Citations**: At the end of your answer, explicitly state the source pages used from DC Dutta. Format it as:
   `📌 **Source Citation**: DC Dutta Obstetrics & Gynecology (Page(s): {pages_ref})`
3. **Accuracy**: Stick strictly to the context provided.

Generate a comprehensive, structured medical response:"""

            # 4. Generate response
            response = llm.invoke(full_prompt)
            answer = response.content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
