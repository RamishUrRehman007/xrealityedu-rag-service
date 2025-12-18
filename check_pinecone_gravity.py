import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

def check_pinecone_content():
    print("--- 🔍 Checking Pinecone Content ---")
    
    # Init Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in .env")
        return

    try:
        pc = Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "xreality-books") # Defaulting to known index name
        index = pc.Index(index_name)
        
        print(f"✅ Connected to Index: {index_name}")
        
        # Init OpenAI for embedding query
        openai_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_key)
        
        query_text = "What is Gravity?"
        print(f"❓ Querying for: '{query_text}'")
        
        embedding = client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Define Filters based on User Request
        filters = {
            "country": "Pakistan",
            "board": "Sindh Board",
            "grade": "Grade-10",
            "subject": "Physics" # Note: RAG implementation usually capitalizes, checked main.py
        }
        
        print(f"🔎 Using Filters: {filters}")
        
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter=filters
        )
        
        if results.matches:
            print(f"✅ Found {len(results.matches)} matches that strictly meet criteria:")
            for m in results.matches:
                meta = m.metadata
                print(f"   - Score: {m.score:.4f}")
                print(f"     Source: {meta.get('source', 'N/A')}") 
                print(f"     File: {meta.get('book_name', 'N/A')}")
                print(f"     Chapter: {meta.get('chapter', 'N/A')}")
                print(f"     Text Snippet: {meta.get('text', '')[:100]}...")
                print("-" * 30)
        else:
            print("❌ No matches found with EXACT filters.")
            
            print("\n🔄 Trying Relaxed Search (removing 'board' and 'country')...")
            relaxed_filters = {
                "grade": "Grade-10",
                "subject": "Physics"
            }
            results_relaxed = index.query(
                vector=embedding,
                top_k=5,
                include_metadata=True,
                filter=relaxed_filters
            )
            
            if results_relaxed.matches:
                 print(f"⚠️ Found {len(results_relaxed.matches)} matches with RELAXED filters (Subject+Grade only):")
                 for m in results_relaxed.matches:
                    print(f"     Metadata: {m.metadata}")
            else:
                 print("❌ No matches found even with relaxed filters.")
            
            print("\n🌍 Performing GLOBAL SEARCH (No Filters) to see what IS indexed...")
            results_global = index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            for m in results_global.matches:
                print(f"   [Score: {m.score:.2f}]")
                print(f"   Metadata: {m.metadata}")
                print("-" * 30)

    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")

if __name__ == "__main__":
    check_pinecone_content()
