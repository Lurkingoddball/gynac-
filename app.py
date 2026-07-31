import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Set clean page config
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

# Initialize Vector DB & LLM directly
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

# Chat memory session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("Ask a question from DC Dutta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching DC Dutta & generating response..."):
            # 1. Retrieve top context
            docs = vector_db.similarity_search(prompt, k=3)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Build prompt directly
            full_prompt = f"""You are an expert AI medical tutor based on DC Dutta's Obstetric & Gynecology textbooks. 
Answer the question accurately, clearly, and concisely based on the retrieved context below.

Context:
{context_text}

Question: {prompt}
"""
            # 3. Get response directly from Groq
            response = llm.invoke(full_prompt)
            answer = response.content
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
