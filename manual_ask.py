from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import json
import re
import asyncio
import random
from dotenv import load_dotenv

from retrieve_and_respond import answer_question
from store_chat_to_pinecone import store_chat_to_pinecone

load_dotenv()

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
welcomed_rooms: Dict[str, bool] = {}
user_chat_count: Dict[str, int] = {}  # Track number of messages per room

# --- AI Style Enhancements ---
thinking_lines = [
    "🧠 AI: Hmm... let me help you with that.",
    "🧠 AI: Let's think this through together.",
    "🧠 AI: Great question! Let me guide you.",
    "🧠 AI: Alright, let's dive in!",
    "🧠 AI: For sure!"
]

subject_emojis = {
    "physics": "📘",
    "math": "📐",
    "biology": "🧬",
    "chemistry": "🧪",
    "general": "📚"
}

def suggest_topic():
    suggestions = [
        "Would you like to learn about Ohm’s Law next?",
        "Want to explore Newton’s Laws of Motion?",
        "Shall we look into the Law of Conservation of Energy?",
        "How about a quick quiz on forces and motion?"
    ]
    return random.choice(suggestions)

@app.websocket("/ws/qa_chat/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    print(f"🟢 Connected to room: {room_id}")

    if room_id not in connections:
        connections[room_id] = []
    if room_id not in chat_logs:
        chat_logs[room_id] = []
    if room_id not in welcomed_rooms:
        welcomed_rooms[room_id] = False
    if room_id not in user_chat_count:
        user_chat_count[room_id] = 0

    connections[room_id].append(websocket)

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)

            user_message = payload["message"]
            user_id = payload["user_id"]
            student_name = payload["student_name"]
            grade = payload["grade"]
            subject = payload["subject"].lower()
            user_chat_count[room_id] += 1

            emoji = subject_emojis.get(subject.lower(), "📚")

            print(f"[{room_id}] {user_id} says: {user_message}")
            print(f"🎓 Student: {student_name} | Grade: {grade} | Subject: {subject}")

            if not welcomed_rooms[room_id]:
                welcome_msg = f"{emoji} Hello {student_name}! I'm Xreality, your friendly AI {subject.capitalize()} Tutor. Let's make learning fun!"
                await websocket.send_json({"message": welcome_msg, "user_id": "AI_TUTOR"})
                welcomed_rooms[room_id] = True

            disallowed_subjects = ["math", "calculus", "algebra", "geometry", "bio", "biology", "chemistry"]
            if any(re.search(word, user_message, re.IGNORECASE) for word in disallowed_subjects):
                if subject.lower() not in user_message.lower():
                    warning_msg = f"🛘 I'm only assisting with **{subject.capitalize()}** in this session. Please switch subject for other topics."
                    await websocket.send_json({"message": warning_msg, "user_id": "AI_TUTOR"})
                    continue

            greetings = ["hi", "hello", "salam", "hey", "good morning", "good evening"]
            is_greeting = any(re.fullmatch(greet, user_message.strip().lower()) for greet in greetings)

            if not is_greeting:
                thinking_message = random.choice(thinking_lines)
                await websocket.send_json({"message": thinking_message, "user_id": "AI_TUTOR"})
                await asyncio.sleep(1.5)

            history = "\n".join(chat_logs[room_id][-6:])
            ai_response = answer_question(
                question=user_message,
                history=history,
                subject=subject,
                student_name=student_name,
                grade_level=grade
            )

            response_text = str(ai_response.content) if hasattr(ai_response, "content") else str(ai_response)

            chat_logs[room_id].append(f"User: {user_message}")
            chat_logs[room_id].append(f"AI: {response_text}")
            chat_logs[room_id] = chat_logs[room_id][-10:]

            await websocket.send_json({"message": response_text, "user_id": "AI_TUTOR"})

            if "Choose A, B, C, or D" in response_text:
                await websocket.send_json({
                    "message": "🧠 I've asked a quiz! Please reply with A, B, C, or D.",
                    "user_id": "AI_TUTOR"
                })

            elif user_chat_count[room_id] >= 3 and any(word in response_text.lower() for word in ["great job", "correct", "let’s continue", "keep it up"]):
                await asyncio.sleep(0.5)
                suggestion = suggest_topic()
                await websocket.send_json({
                    "message": f"🧠 {suggestion}",
                    "user_id": "AI_TUTOR"
                })

    except WebSocketDisconnect:
        print(f"🔴 Disconnected from room: {room_id}")

        if chat_logs.get(room_id):
            print(f"🧠 Saving chat history for: {room_id}")
            print("📄 Chat Log to Save:", chat_logs[room_id])

            result = store_chat_to_pinecone(
                chat_history=chat_logs[room_id],
                student=user_id,
                grade=grade
            )
            print(f"📦 Chat stored to Pinecone: {result}")

            try:
                await websocket.send_json({
                    "message": "✅ Your session has ended and the chat was saved successfully.",
                    "user_id": "SYSTEM"
                })
            except:
                pass

        if room_id in connections:
            connections[room_id].remove(websocket)

        if not connections[room_id]:
            chat_logs.pop(room_id, None)
            connections.pop(room_id, None)
            welcomed_rooms.pop(room_id, None)
            user_chat_count.pop(room_id, None)
