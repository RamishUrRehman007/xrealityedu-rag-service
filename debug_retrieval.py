import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load params
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def debug_retrieval():
    print("--- Debugging Pinecone Retrieval ---")
    
    # 1. Connect
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to Index: {PINECONE_INDEX_NAME}")

    # 2. Generate Embedding for Query
    query_text = "Gravity Concepts"
    print(f"Query: '{query_text}'")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    res = client.embeddings.create(input=query_text, model="text-embedding-ada-002")
    embed = res.data[0].embedding

    # 3. Search WITHOUT Filters (to see what exists)
    print("\n[Test 1] Searching WITHOUT filters (Top 3):")
    results = index.query(vector=embed, top_k=3, include_metadata=True)
    
    for m in results['matches']:
        print(f"  Score: {m['score']:.4f}")
        print(f"  Metadata: {m['metadata']}")
        print("-" * 20)

    # 4. Search WITH Filters (Simulating the failure)
    # Filter used by RAG: board="Pakistan - Sindh - Sindh Board" ?
    # Let's try to match what main.py uses.
    target_filter = {
        "subject": "Physics",
        "grade": "Grade-10",
        # Loose match logic might be needed if exact match fails
    }
    
    print(f"\n[Test 2] Searching WITH Strict Filter: {target_filter}")
    try:
        results_filtered = index.query(
            vector=embed, 
            top_k=3, 
            include_metadata=True, 
            filter=target_filter
        )
        if not results_filtered['matches']:
            print("  ❌ No matches found with filter.")
        else:
            for m in results_filtered['matches']:
                print(f"  Score: {m['score']:.4f}")
                print(f"  Metadata: {m['metadata']}")
    except Exception as e:
        print(f"Filter Error: {e}")

if __name__ == "__main__":
    debug_retrieval()
