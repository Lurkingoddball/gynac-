import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Configuration
st.set_page_config(
    page_title="DC Dutta Medical Tutor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- GEMINI UI STYLING (CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #131314;
        color: #E3E3E3;
        font-family: 'Google Sans', -apple-system, sans-serif;
    }
    .gemini-title {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(90deg, #4285F4, #9B51E0, #E91E63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .gemini-badge {
        background-color: #1E1F20;
        border: 1px solid #333537;
        color: #C4C7C5;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #1E1F20 !important;
        border: 1px solid #28292A !important;
        border-radius: 18px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Header
st.markdown('<div class="gemini-title">✨ DC Dutta OB/GYN AI</div>', unsafe_allow_html=True)
st.markdown('<div class="gemini-badge">🟢 Cloud Active • Always-On Access</div>', unsafe_allow_html=True)

# Fetch API Key securely
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ Groq API key missing. Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# Initialize RAG Chain
@st.cache_resource
def load_rag_chain():
    VECTOR_DB_DIR = "./dutta_vector_db"
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_function)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    system_prompt = """
    You are an AI Medical Tutor grounded strictly in DC Dutta's Textbooks.
    
    INSTRUCTIONS:
    1. Answer using ONLY provided context.
    2. Add inline citations for key points, e.g., [DC Dutta Obstetrics, p. 159].
    3. Format cleanly with bullet points, bold headings, and clear clinical structure.
    4. Guardrail: If answer is not in context, state: "I cannot find sufficient details on this in DC Dutta textbooks."

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        formatted = []
        for doc in docs:
            subject = doc.metadata.get("subject", "DC Dutta Textbook")
            page = doc.metadata.get("page", 0) + 1
            formatted.append(f"--- SOURCE: DC Dutta {subject} (Page {page}) ---\n{doc.page_content.strip()}\n")
        return "\n".join(formatted)

    # Cloud LLM powered by Groq LPU
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.2
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Error initializing RAG pipeline: {e}")
    st.stop()

# Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your DC Dutta AI Medical Tutor. Ask any question from Obstetrics or Gynecology."}
    ]

for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Prompt Input
if prompt_input := st.chat_input("Ask anything about OB/GYN..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Searching DC Dutta textbooks..."):
            try:
                response = rag_chain.invoke(prompt_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Generation error: {e}")