import os
import uuid
import re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

load_dotenv()

def _redact_self_harm(text: str) -> str:
    if not text:
        return text
    patterns = [
        r"\bkill myself\b",
        r"\bi want to die\b",
        r"\bsuicide\b",
        r"\bself[-\s]?harm\b",
        r"\bhurt myself\b",
        r"\bend my life\b",
    ]
    redacted = text
    for p in patterns:
        redacted = re.sub(p, "[self-harm phrase]", redacted, flags=re.IGNORECASE)
    return redacted

def store_chat_to_pinecone(
    chat_history,
    student="Student",
    grade="",
    room_id="",
    subject="physics",
    skip_store: bool = False,
    event_type: str = "interaction",   # "interaction" | "safety_event"
    safety_labels: list = None,
):
    """
    Stores tutoring interactions to Pinecone.
    For safety events, store minimal/redacted text and put it in a separate namespace.
    """
    if skip_store:
        return {"status": "skipped"}

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

    # Keep it small
    last_lines = chat_history[-10:]
    combined_text = "\n".join(last_lines)

    # Safety event: minimize what we store
    namespace = "default"
    store_text = combined_text

    if event_type == "safety_event":
        namespace = "safety"
        # store only last 2 lines, redacted
        store_text = "\n".join(chat_history[-2:])
        store_text = _redact_self_harm(store_text)

    try:
        response = openai_client.embeddings.create(
            input=[store_text],
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
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "text": store_text,
            "subject": subject.lower(),
            "safety_labels": safety_labels or []
        }
    }

    try:
        index.upsert(vectors=[vector], namespace=namespace)
        return {"status": "stored", "chunks": 1, "student": student, "grade": grade, "type": event_type, "namespace": namespace}
    except Exception as e:
        return {"status": "upsert failed", "error": str(e)}