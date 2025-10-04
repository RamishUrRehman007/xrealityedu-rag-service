# ✅ chat_embedder.py
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ENV variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Function to store user-AI chat history into Pinecone
def store_chat_to_pinecone(user_input, ai_response, student_name="Saad"):
    combined_text = f"User: {user_input}\nAI: {ai_response}"

    # Generate embedding
    response = openai_client.embeddings.create(
        input=combined_text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    # Build vector payload
    vector = {
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "source": "chat-history",
            "student": student_name,
            "type": "interaction",
            "timestamp": datetime.now().isoformat()
        }
    }

    # Upsert to Pinecone
    index.upsert(vectors=[vector])
    print("✅ Chat interaction stored in Pinecone.")
