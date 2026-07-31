import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Set page config
st.set_page_config(
    page_title="DC Dutta Medical AI",
    page_icon="🩺",
    layout="centered"
)

st.title("DC Dutta Medical AI")

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

# User prompt
if prompt := st.chat_input("Ask anything from Gyanc-Obs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing DC Dutta & generating detailed response..."):
            # 1. Retrieve relevant textbook context
            docs = vector_db.similarity_search(prompt, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # 2. Format recent conversation history (last 6 turns for memory context)
            chat_history_str = ""
            recent_messages = st.session_state.messages[-6:-1]  # Exclude current prompt
            for msg in recent_messages:
                role_label = "Student" if msg["role"] == "user" else "Tutor"
                chat_history_str += f"{role_label}: {msg['content']}\n"

            if not chat_history_str:
                chat_history_str = "None (This is the start of the conversation)."

            # 3. Comprehensive NotebookLM-style System Prompt
            full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology, trained on DC Dutta's Textbooks.
Your objective is to provide structured, highly detailed, medical-school grade explanations similar to NotebookLM or clinical board review notes.

=== RECENT CONVERSATION HISTORY ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. **Maintain Continuity**: Use the recent conversation history to understand follow-up questions, pronouns (it, this, that), or requests for clarification.
2. **In-Depth Explanation**: Provide thorough, structured answers. Do not summarize in 1-2 brief sentences unless explicitly asked.
3. **Structured Formatting**: Use bolding, clear headings, bullet points, and numbered lists where clinical steps or management protocols are involved.
4. **Clinical Accuracy**: Include definitions, etiology, clinical features, diagnostic criteria, and line-of-treatment where applicable, citing DC Dutta context.

Generate a comprehensive, beautifully structured medical response:"""

            # 4. Generate response from Groq LPU
            response = llm.invoke(full_prompt)
            answer = response.content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
