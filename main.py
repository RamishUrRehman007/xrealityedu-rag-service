# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import asyncio
import os
from functools import partial
from dotenv import load_dotenv

from retrieve_and_respond import answer_question, get_user_chat_history, suggest_topics_with_ai, generate_prompt_suggestions, check_question_similarity
from store_chat_to_pinecone import store_chat_to_pinecone
from embed import embed_pdf

load_dotenv()

# Configure FastAPI with larger file upload limit
# Default Starlette limit is 1MB for request body
# You can set MAX_FILE_SIZE_MB in .env (default: 500MB)
# Note: If using nginx as reverse proxy, configure client_max_body_size in nginx config
# The endpoint validates file size and returns a clear error if exceeded (default: 500MB)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connections: Dict[str, List[WebSocket]] = {}
chat_logs: Dict[str, List[str]] = {}
welcome_sent: Dict[str, bool] = {}
tutor_state: Dict[str, Dict] = {}
question_history: Dict[str, List[str]] = {}  # Track questions per room for repeat detection

@app.websocket("/wss/qa_chat/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    print(f"🟢 Connected to room: {room_id}")

    if room_id not in connections:
        connections[room_id] = []
    if room_id not in chat_logs:
        chat_logs[room_id] = []
    if room_id not in welcome_sent:
        welcome_sent[room_id] = False
    if room_id not in tutor_state:
        tutor_state[room_id] = {
            "step": "awaiting_topic",
            "topic": None,
            "level": None,
            "prior_knowledge": None,
            "quiz_permission": False
        }
    if room_id not in question_history:
        question_history[room_id] = []

    connections[room_id].append(websocket)

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)

            user_message = payload["message"].strip()
            user_id = payload["user_id"]
            student_name = payload["student_name"]
            grade = payload["grade"]
            subject = payload["subject"]

            print(f"[{room_id}] {user_id} says: {user_message}")
            print(f"🎓 Student: {student_name} | Grade: {grade} | Subject: {subject}")

            state = tutor_state[room_id]

            if not welcome_sent[room_id]:
                welcome_sent[room_id] = True

                # Load chat history for this user and subject
                history_entries = get_user_chat_history(user_id, subject)
                
                if len(history_entries) > 0:
                    # Load previous chat history into current session
                    for entry in history_entries[-10:]:  # Load last 10 interactions
                        chat_logs[room_id].append(entry)
                    
                    # If history exists, continue directly
                    state["step"] = "tutoring"
                    state["quiz_permission"] = True
                    await websocket.send_json({
                        "message": f"📚 Welcome back {student_name}! I've loaded your past session with **{subject}**. Let's continue where we left off!",
                        "user_id": "AI_TUTOR"
                    })
                    
                    # Show recent conversation context
                    if len(history_entries) > 0:
                        recent_context = history_entries[-2] if len(history_entries) >= 2 else history_entries[-1]
                        await websocket.send_json({
                            "message": f"💭 Last time we discussed: {recent_context[:100]}...",
                            "user_id": "AI_TUTOR"
                        })
                    continue

                # Otherwise, fresh start
                await websocket.send_json({
                    "message": f"👋 Hi {student_name}, I'm your AI tutor and I'm excited to help you today!\nWhat would you like to learn about in **{subject}**?",
                    "user_id": "AI_TUTOR"
                })

                topics = suggest_topics_with_ai(subject, grade, student_name, student_id=user_id)
                await websocket.send_json({
                    "message": f"📘 Here are some topics you might explore in **{subject}**:",
                    "topics": topics,
                    "type": "topic_suggestions",
                    "user_id": "AI_TUTOR"
                })
                continue

            # Tutor state logic
            if state["step"] == "awaiting_topic":
                state["topic"] = user_message
                # Infer level from grade (already provided in payload)
                # Grade format: "Grade 10", "Grade 12", etc.
                if "grade" in grade.lower() or any(g in grade.lower() for g in ["9", "10", "11", "12"]):
                    state["level"] = "high school student"
                elif any(g in grade.lower() for g in ["college", "university", "undergrad"]):
                    state["level"] = "college student"
                else:
                    state["level"] = grade  # Use grade as-is if format is different
                
                state["step"] = "awaiting_prior_knowledge"
                await websocket.send_json({
                    "message": f"Great! What do you already know about **{state['topic']}**?",
                    "user_id": "AI_TUTOR"
                })
                continue

            elif state["step"] == "awaiting_prior_knowledge":
                state["prior_knowledge"] = user_message
                state["step"] = "tutoring"
                await websocket.send_json({
                    "message": f"Awesome. Let's dive into **{state['topic']}** together!",
                    "user_id": "AI_TUTOR"
                })
                user_message = f"Explain {state['topic']} to a {state['level']} learner who already knows: {state['prior_knowledge']}"

            # Handle unsure students
            if user_message.lower() in ["i don't know", "not sure"]:
                await websocket.send_json({
                    "message": "No worries! If you'd like me to guide you step by step, just say **yes**. I’ll also give some real-life examples to help you understand.",
                    "user_id": "AI_TUTOR"
                })
                continue

            if user_message.lower() in ["yes", "okay", "sure"] and state["step"] == "tutoring" and not state["quiz_permission"]:
                state["quiz_permission"] = True
                await websocket.send_json({
                    "message": "Great! Would you like to try a short quiz on what we've discussed so far?",
                    "user_id": "AI_TUTOR"
                })
                continue

            if user_message.lower() in ["yes quiz", "start quiz", "quiz", "let's do quiz"] and state["quiz_permission"]:
                history = "\n".join(chat_logs[room_id][-6:])
                loop = asyncio.get_event_loop()
                quiz_response = await loop.run_in_executor(
                    None,
                    partial(
                        answer_question,
                        question=user_message,
                        history=history,
                        subject=subject,
                        student_name=student_name,
                        grade_level=grade,
                        mode="quiz"
                    )
                )
                await websocket.send_json({"message": quiz_response, "user_id": "AI_TUTOR"})
                continue

            # Prevent subject mismatch
            subjects = ["physics", "math", "biology", "chemistry", "english", "urdu", "islamiyat", "computer"]
            normalized_msg = user_message.lower()
            if any(s in normalized_msg and s != subject.lower() for s in subjects):
                await websocket.send_json({
                    "message": f"🛛 This seems related to another subject. You're currently in **{subject}** session.",
                    "user_id": "AI_TUTOR"
                })
                continue

            # Check for repeated similar questions (only in tutoring mode)
            if state["step"] == "tutoring":
                previous_questions = question_history.get(room_id, [])
                if len(previous_questions) >= 2:  # Need at least 2 previous questions to check
                    is_similar, similar_count, most_similar_q = check_question_similarity(
                        user_message, 
                        previous_questions,
                        similarity_threshold=0.85
                    )
                    
                    # If this question is similar to 2+ previous questions, suggest moving on
                    if is_similar and similar_count >= 2:  
                        # Get topic suggestions for next topic
                        topics = suggest_topics_with_ai(subject, grade, student_name, student_id=user_id)
                        await websocket.send_json({
                            "message": f"🔄 I notice you've asked about this topic a few times. It might be helpful to move on to a new topic to keep learning fresh! Let's explore something new in **{subject}**.",
                            "user_id": "AI_TUTOR"
                        })
                        await websocket.send_json({
                            "message": f"📘 Here are some new topics you might explore in **{subject}**:",
                            "topics": topics,
                            "type": "topic_suggestions",
                            "user_id": "AI_TUTOR"
                        })
                        # Clear question history to start fresh with new topic
                        question_history[room_id] = []
                        continue
                    elif is_similar:
                        # First or second similar question - still answer but track it
                        print(f"⚠️ Similar question detected ({similar_count} similar questions found). Most similar: {most_similar_q}")

            # Track the question
            question_history[room_id].append(user_message)
            # Keep only last 20 questions to avoid memory issues
            question_history[room_id] = question_history[room_id][-20:]

            # Regular tutoring flow
            history = "\n".join(chat_logs[room_id][-6:])
            loop = asyncio.get_event_loop()
            ai_response = await loop.run_in_executor(
                None,
                partial(
                    answer_question,
                    question=user_message,
                    history=history,
                    subject=subject,
                    student_name=student_name,
                    grade_level=grade,
                    mode="tutoring"
                )
            )

            response_text = str(ai_response.content) if hasattr(ai_response, "content") else str(ai_response)
            chat_logs[room_id].append(f"User: {user_message}")
            chat_logs[room_id].append(f"AI: {response_text}")
            chat_logs[room_id] = chat_logs[room_id][-10:]

            # Generate prompt suggestions for the response
            try:
                suggestions = generate_prompt_suggestions(
                    question=user_message,
                    response=response_text,
                    subject=subject,
                    student_name=student_name,
                    grade_level=grade,
                    history=history
                )
            except Exception as e:
                print(f"⚠️ Error generating suggestions: {e}")
                suggestions = []

            # Send the main response
            await websocket.send_json({"message": response_text, "user_id": "AI_TUTOR"})
            
            # Send prompt suggestions if available (only for regular conversations, not quizzes)
            if suggestions and len(suggestions) > 0 and "Choose A, B, C, or D" not in response_text:
                await asyncio.sleep(0.5)  # Small delay for better UX
                await websocket.send_json({
                    "message": "💡 Here are some follow-up questions you might ask:",
                    "suggestions": suggestions,
                    "type": "prompt_suggestions",
                    "user_id": "AI_TUTOR"
                })

    except WebSocketDisconnect:
        print(f"🔴 Disconnected from room: {room_id}")

        if chat_logs.get(room_id):
            print(f"🧠 Saving chat history for: {room_id}")
            result = store_chat_to_pinecone(
                chat_history=chat_logs[room_id],
                student=user_id,
                grade=grade,
                room_id=room_id,
                subject=subject
            )
            print(f"📦 Chat stored to Pinecone: {result}")

        for store in [connections, chat_logs, welcome_sent, tutor_state, question_history]:
            store.pop(room_id, None)

# ============================================================================
# FILE UPLOAD ENDPOINT FOR RAG EMBEDDING
# ============================================================================

@app.post("/api/upload")
async def upload_file_for_embedding(
    file: UploadFile = File(...),
    subject: str = Form(...),
    grade: Optional[str] = Form(None),
    curriculum: Optional[str] = Form(None),
    source: Optional[str] = Form(None)
):
    """
    Upload and embed files (PDF, TXT, etc.) for RAG system.
    
    Parameters:
    - file: The file to upload (PDF, TXT, MD, etc.)
    - subject: Subject name (required, e.g., "Physics", "Math", "Biology")
    - grade: Grade level (optional, e.g., "Grade 10")
    - curriculum: Curriculum name (optional)
    - source: Source identifier (optional, e.g., "physics30-A")
    
    Returns:
    - Status and number of chunks uploaded to Pinecone
    """
    try:
        # Validate file and filename
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.md']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if not file_extension or file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Get file size limit from environment (default: 500MB)
        MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "500")) * 1024 * 1024  # Convert MB to bytes
        
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        file_size = len(contents)
        if file_size > MAX_FILE_SIZE:
            max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=413,  # 413 Payload Too Large
                detail=f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({max_size_mb:.0f} MB). Please upload a smaller file."
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file
        filename = file.filename
        save_path = f"uploads/{filename}"
        
        with open(save_path, "wb") as f:
            f.write(contents)
        with open(save_path, "wb") as f:
            f.write(contents)
        
        print(f"📄 File uploaded: {filename} ({len(contents)} bytes)")
        
        # Prepare metadata
        metadata = {
            "subject": subject.capitalize(),  # Capitalize for consistency with existing data
            "source": source or os.path.splitext(filename)[0],  # Use filename without extension
        }
        
        if grade:
            metadata["grade"] = grade
        if curriculum:
            metadata["curriculum"] = curriculum
        
        # Handle different file types
        if file_extension == '.pdf':
            # Use existing PDF embedding function
            result = embed_pdf(save_path, metadata)
        elif file_extension in ['.txt', '.md']:
            # Handle text files
            result = embed_text_file(save_path, metadata)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} processing not yet implemented"
            )
        
        return {
            "status": "success",
            "filename": filename,
            "file_size": len(contents),
            "chunks_uploaded": result.get("uploaded_chunks", 0),
            "subject": subject,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

def embed_text_file(file_path: str, metadata: dict):
    """
    Embed text files (TXT, MD) into Pinecone.
    Similar to embed_pdf but for text files.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec
    import uuid
    
    # Read text file with error handling for encoding
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Failed to read file with UTF-8 or Latin-1 encoding: {str(e)}")
    
    if not content or len(content.strip()) == 0:
        raise Exception("File is empty or contains no text content")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(content)
    
    if not texts or len(texts) == 0:
        raise Exception("No text chunks could be extracted from the file")
    
    # Prepare metadata for each chunk
    metadatas = [metadata.copy() for _ in texts]
    
    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_ENVIRONMENT")
            )
        )
    
    index = pc.Index(index_name)
    
    # Constants
    MAX_PAYLOAD_BYTES = 1_800_000
    BYTES_PER_FLOAT32 = 4
    VECTOR_BYTES = 1536 * BYTES_PER_FLOAT32
    
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
            "metadata": {
                **meta,
                "text": text  # Store the actual text content
            }
        } for embedding, meta, text in zip(embeddings, batch_metas, batch_texts)]
        
        index.upsert(vectors=vectors)
        uploaded += len(vectors)
        
        batch_texts, batch_metas, batch_bytes = [], [], 0
    
    # Process in batches
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
