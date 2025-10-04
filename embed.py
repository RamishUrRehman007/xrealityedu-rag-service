import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Constants
MAX_PAYLOAD_BYTES = 1_800_000  # Pinecone limit is 4MB. Keep some buffer
EMBED_DIM = 1536
BYTES_PER_FLOAT32 = 4
VECTOR_BYTES = EMBED_DIM * BYTES_PER_FLOAT32

def embed_pdf(file_path, metadata):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)

    for doc in docs:
        doc.metadata.update(metadata)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_ENVIRONMENT")
            )
        )

    index = pc.Index(index_name)

    batch_texts = []
    batch_metas = []
    batch_bytes = 0
    uploaded = 0

    def upsert():
        nonlocal uploaded, batch_texts, batch_metas, batch_bytes
        if not batch_texts:
            return

        response = openai_client.embeddings.create(
            input=batch_texts,
            model="text-embedding-3-small"
        )
        embeddings = [e.embedding for e in response.data]

        vectors = [{
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": meta
        } for embedding, meta in zip(embeddings, batch_metas)]

        index.upsert(vectors=vectors)
        uploaded += len(vectors)

        batch_texts, batch_metas, batch_bytes = [], [], 0

    for text, meta in zip(texts, metadatas):
        text_bytes = len(text.encode("utf-8"))
        record_bytes = text_bytes + VECTOR_BYTES
        if batch_bytes + record_bytes >= MAX_PAYLOAD_BYTES:
            upsert()
        batch_texts.append(text)
        batch_metas.append(meta)
        batch_bytes += record_bytes

    upsert()

    return {"status": "success", "uploaded_chunks": uploaded}

# Run example
if __name__ == "__main__":
    result = embed_pdf("uploads/physics30-B.pdf", {"source": "physics30-B"})
    print(result)
