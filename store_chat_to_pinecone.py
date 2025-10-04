import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

load_dotenv()

def store_chat_to_pinecone(chat_history, student="Student", grade="", room_id=""):
    if not chat_history:
        return {"status": "no chat to store"}

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    combined_text = "\n".join(chat_history[-10:])

    try:
        response = openai_client.embeddings.create(
            input=[combined_text],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
    except Exception as e:
        return {"status": "embedding failed", "error": str(e)}

    vector = {
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "source": "chat-history",
            "student": student,
            "grade": grade,
            "room_id": room_id,
            "type": "interaction",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    try:
        index.upsert(vectors=[vector])
        return {"status": "stored", "chunks": 1, "student": student, "grade": grade}
    except Exception as e:
        return {"status": "upsert failed", "error": str(e)}