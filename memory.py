import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import warnings

# Suppress some noisy warnings from the embedding library
warnings.filterwarnings("ignore")

print("Initializing Agent Memory Core (Vector DB)...")

# 1. Initialize the embedding model (This runs locally on your Mac!)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Set up the local Vector Database
client = chromadb.PersistentClient(path="./agent_memory_db")
collection = client.get_or_create_collection(name="research_findings")

def store_finding(summary: str, topic: str):
    """Embeds a text finding and saves it to long-term memory."""
    # Create a unique ID for this finding
    doc_id = hashlib.md5(summary.encode()).hexdigest()
    
    # Convert the text into a mathematical vector
    embedding = embedder.encode(summary).tolist()
    
    # Save it to ChromaDB
    collection.add(
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{"topic": topic}],
        ids=[doc_id]
    )

def recall_relevant(query: str, n_results: int = 2):
    """Searches memory for similar past findings."""
    embedding = embedder.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    
    # Return the documents if the database isn't empty
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0]
    return []

# --- Quick Test ---
if __name__ == "__main__":
    print("\n[Test] Storing a fact in memory...")
    store_finding("Anthropic recently raised funding at a $900 billion valuation.", "Anthropic Funding")
    
    print("[Test] Asking the database a question...")
    memories = recall_relevant("How much is Anthropic worth?")
    
    print("\n--- Recalled Memory ---")
    for m in memories:
        print(f"- {m}")