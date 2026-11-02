from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def search_database(query: str, k: int = 2):
    # 1. Load the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Re-connect to your persistent Chroma database
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # 3. Convert query to vector and find top 'k' closest matches
    # 'similarity_search_with_score' returns the chunk and its distance score
    results = vector_db.similarity_search_with_score(query, k=k)
    
    print(f"\n================ QUERY: '{query}' ================")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n--- Match #{i} (Distance Score: {score:.4f}) ---")
        print(f"Source Document: {doc.metadata.get('source')}")
        print("Content:")
        print(doc.page_content)
        print("=" * 55)

if __name__ == "__main__":
    # Test queries targeting your three loaded files
    search_database("How do I configure S/MIME for git on Linux?", k=2)
    search_database("What is an instance of a class in Python?", k=1)
    search_database("How do I run a Docker container in interactive mode?", k=1)