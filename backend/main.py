import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

app = FastAPI(title="Gynaec-Obs AI Backend", version="2.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite Database Initialization
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

init_db()

# Load Embeddings, Vector Store & Groq LLM
groq_api_key = os.environ.get("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="../dutta_vector_db",
    embedding_function=embeddings
)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# Request & Response Schemas
class ChatRequest(BaseModel):
    session_id: str
    prompt: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set.")

    session_id = req.session_id
    prompt = req.prompt.strip()

    # 1. Save user query to database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, "user", prompt))
    
    # 2. Retrieve recent conversation turns for context memory
    c.execute("SELECT role, content FROM history WHERE session_id = ? ORDER BY id DESC LIMIT 6", (session_id,))
    recent_rows = c.fetchall()[::-1]
    conn.close()

    recent_queries = [r[1] for r in recent_rows if r[0] == "user"]
    search_query = f"{recent_queries[-2]} {prompt}" if len(recent_queries) > 1 else prompt

    chat_history_str = ""
    for r in recent_rows:
        label = "Student" if r[0] == "user" else "Tutor"
        chat_history_str += f"{label}: {r[1]}\n"

    # 3. Retrieve relevant chunks from ChromaDB
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

    # 4. System Prompt Generation
    full_prompt = f"""You are an elite Medical AI Tutor specialized in Obstetrics and Gynecology based strictly on DC Dutta's Textbook.
Provide highly detailed, medical-school grade explanations with explicit clinical steps.

=== CONVERSATION HISTORY (FOR CONTINUITY) ===
{chat_history_str}

=== RETRIEVED TEXTBOOK CONTEXT ===
{context_text}

=== CURRENT QUESTION ===
{prompt}

=== INSTRUCTIONS FOR RESPONSE ===
1. Maintain continuity for follow-up questions.
2. Structure with standard medical headings:
   - **Definition / Overview**
   - **Etiology & Pathophysiology**
   - **Clinical Features & Diagnosis**
   - **Management / Line of Treatment** (Medical, Surgical, Emergency)
3. Include bullet points, bold key medical terms, and dosage protocols.
4. End strictly with: `{citation_line}`"""

    response = llm.invoke(full_prompt)
    answer = response.content

    # Save Assistant response to database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, "assistant", answer))
    conn.commit()
    conn.close()

    return ChatResponse(session_id=session_id, answer=answer, citations=citation_line)

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM history WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]
