import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables (.env)
load_dotenv()

# 1. Initialize Direct Clients
groq_client = Groq()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Access persistent Chroma collection
collection = chroma_client.get_collection(name="langchain")

def ask_question(query: str):
    # Step A: Vectorize query and search ChromaDB directly
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2
    )
    
    # Step B: Extract matching text chunks and metadata
    retrieved_texts = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    
    context = "\n\n".join(retrieved_texts)
    
    # Step C: Format System/User Prompt
    prompt = f"""
You are a technical research software engineering assistant.
Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer, state clearly: "I cannot find this information in the technical documentation."

Context:
{context}

Question: {query}
"""

    # Step D: Call Groq API with an active model ID
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    # Display Output
    print(f"\n[User Query]: {query}")
    print("-" * 50)
    print("[Groq LLM Response]:")
    print(response.choices[0].message.content)
    print("\n[Retrieved Sources Used]:")
    for meta in retrieved_metadatas:
        print(f" - {meta.get('source', 'Unknown')}")
    print("=" * 50)

if __name__ == "__main__":
    ask_question("How do I configure S/MIME commit signing on Linux?")
    ask_question("What is an object instance in Python and how do we construct it?")
    ask_question("How do I execute commands inside a running Docker container?")