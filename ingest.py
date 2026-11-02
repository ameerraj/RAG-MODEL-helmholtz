from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def process_and_store_documents():
    print("1. Loading documents...")
    loader = DirectoryLoader(
        './docs', 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    print("2. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    print("3. Generating embeddings & saving to ChromaDB...")
    # Downloads a tiny local embedding model on first run
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store vectors in a local folder called ./chroma_db
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Success! Chunks and vectors are now stored permanently in './chroma_db'.")

if __name__ == "__main__":
    process_and_store_documents()