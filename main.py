# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import json
import asyncio
from functools import partial
from dotenv import load_dotenv

from retrieve_and_respond import answer_question, get_user_chat_history, suggest_topics_with_ai, generate_prompt_suggestions, check_question_similarity
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
                state["step"] = "awaiting_level"
                await websocket.send_json({
                    "message": "Great! Are you a high school student, college student, or a professional?",
                    "user_id": "AI_TUTOR"
                })
                continue

            elif state["step"] == "awaiting_level":
                state["level"] = user_message.lower()
                state["step"] = "awaiting_prior_knowledge"
                await websocket.send_json({
                    "message": f"Thanks! What do you already know about **{state['topic']}**?",
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
