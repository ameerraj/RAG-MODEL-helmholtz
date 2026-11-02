import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HIFIS Chat API")

# Initialize RAG components on startup
groq_client = Groq()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="langchain")

class QueryRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Serve index.html directly at http://localhost:8000
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
def chat_endpoint(request: QueryRequest):
    # 1. Vector Search in ChromaDB
    query_vector = embedding_model.encode(request.prompt).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2
    )
    
    retrieved_texts = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    context = "\n\n".join(retrieved_texts)

    # 2. Construct Prompt
    system_prompt = f"""
You are a technical research software engineering assistant for HIFIS.
Answer the user's question using ONLY the provided context below.
If the context does not contain enough information, state: "I cannot find this information in the technical documentation."

Context:
{context}

Question: {request.prompt}
"""

    # 3. Call Groq
    response = groq_client.chat.completions.create(
        # Note: If 'openai/gpt-oss-120b' throws an error, switch to a valid Groq model like 'llama-3.1-8b-instant'
        model="openai/gpt-oss-120b", 
        messages=[{"role": "user", "content": system_prompt}],
        temperature=0.0
    )

    answer = response.choices[0].message.content
    sources = [meta.get('source', 'Unknown') for meta in retrieved_metadatas]

    return {
        "answer": answer,
        "sources": sources,
        "snippets": retrieved_texts
    }

# This block ensures the server runs when you execute 'python app.py'
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)