# main.py
import sys
if sys.version_info < (3, 10):
    try:
        import importlib.metadata
        import importlib_metadata
        if not hasattr(importlib.metadata, "packages_distributions"):
            importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
    except ImportError:
        pass  # importlib-metadata not installed, hope for the best

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import asyncio
import re
import os
from functools import partial
from dotenv import load_dotenv
import time # ✅ Added time
import requests # ✅ Added requests
import random # For lesson runner
import traceback # For error logging
import retrieve_and_respond
from teaching_pack import generate_teaching_pack

from retrieve_and_respond import answer_question, get_user_chat_history, suggest_topics_with_ai, generate_prompt_suggestions, check_question_similarity, handle_chip_click
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

# Mount static directory for generated images
STATIC_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(STATIC_IMAGE_DIR):
    os.makedirs(STATIC_IMAGE_DIR)

app.mount("/uploads", StaticFiles(directory=STATIC_IMAGE_DIR), name="uploads")

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

# Backend API URL for progress updates
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:5001/api")
ALERT_THROTTLE_SECONDS = int(os.getenv("ALERT_THROTTLE_SECONDS", "1800"))

# Track last heartbeat time per user to avoid spamming
last_heartbeat: Dict[str, float] = {}
HEARTBEAT_INTERVAL = 60  # Send heartbeat every 60 seconds (1 minute)

def send_study_heartbeat(user_id: str, minutes: int = 1):
    """
    Send heartbeat to backend to track study time.
    Runs in background thread to not block response.
    """
    import threading
    
    def _do_heartbeat():
        try:
            import requests
            import time
            
            # Rate limit heartbeats
            current_time = time.time()
            if user_id in last_heartbeat:
                elapsed = current_time - last_heartbeat[user_id]
                if elapsed < HEARTBEAT_INTERVAL:
                    return  # Too soon, skip this heartbeat
            
            last_heartbeat[user_id] = current_time
            
            payload = {
                "userId": user_id,
                "minutes": minutes
            }
            response = requests.post(
                f"{BACKEND_API_URL}/internal/study/heartbeat",
                json=payload,
                timeout=3  # Reduced timeout
            )
            if response.status_code == 200:
                data = response.json()
                print(f"⏱️ Study heartbeat: {data.get('todayMinutes', 0)} mins today")
            else:
                print(f"⚠️ Heartbeat failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
    
    # Run in background thread
    threading.Thread(target=_do_heartbeat, daemon=True).start()

async def study_heartbeat_loop(user_id: str, room_id: str):
    """
    Periodically send study heartbeat while websocket is active.
    """
    print(f"⏱️ Starting study timer for: {user_id} (Connected)")
    try:
        while True:
            # Check if connection is still active (using the global connections dict)
            # Use a deeper check if needed, but room_id existence is a good proxy 
            # (connection is removed from dict on disconnect)
            if room_id not in connections or not connections[room_id]:
                print(f"🛑 Stopped study timer for: {user_id} (Disconnected)")
                break
                
            # Send heartbeat (1 minute credit)
            send_study_heartbeat(user_id, minutes=1)
            
            # Wait 60 seconds
            await asyncio.sleep(60)
            
    except asyncio.CancelledError:
        print(f"🛑 Study timer cancelled for: {user_id}")
    except Exception as e:
        print(f"⚠️ Study timer error: {e}")

def update_chapter_progress(user_id: str, subject: str, topic_title: str = None, chapter_title: str = None):
    """
    Call Node.js backend to update chapter progress when a topic is started.
    Runs in background thread to not block response.
    """
    import threading
    
    def _do_update():
        try:
            import requests
            payload = {
                "userId": user_id,
                "subject": subject,
                "topicTitle": topic_title,
                "chapterTitle": chapter_title
            }
            response = requests.post(
                f"{BACKEND_API_URL}/internal/progress/start-topic",
                json=payload,
                timeout=3  # Reduced timeout
            )
            if response.status_code == 200:
                print(f"📊 Progress updated: {subject} - {topic_title or chapter_title}")
            else:
                print(f"⚠️ Progress update failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Progress update error: {e}")
    
    # Run in background thread
    threading.Thread(target=_do_update, daemon=True).start()

def fetch_teaching_pack(subject: str, chapter_title: str) -> dict:
    """
    Fetch Teaching Pack JSON from Node.js Backend.
    """
    try:
        import requests
        # Filter is tricky if we don't have chapterId. Try subject+title.
        # Ensure encoding is handled
        url = f"{BACKEND_API_URL}/internal/chapters-teaching-pack"
        params = {
            "subject": subject,
            "chapterTitle": chapter_title
        }
        print(f"📦 Fetching Teaching Pack: {subject} - {chapter_title}...")
        resp = requests.get(url, params=params, timeout=5)
        
        if resp.status_code == 200:
            pack = resp.json()
            # print(f"✅ Teaching Pack Loaded: {pack.keys()}")
            return pack
        else:
            print(f"⚠️ Teaching Pack fetch failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Error fetching Teaching Pack: {e}")
        return None


# ============================================================================
# HELPER FUNCTIONS: Crisis, Pack Fetching, Chapter Resolution, Lesson Runner
# ============================================================================

def detect_self_harm_strong(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "suicide","kill myself","i want to die","self harm","self-harm","hurt myself","end my life"
    ])

def trigger_parent_alert_via_node(user_id: str, message: str, subject: str):
    try:
        requests.post(
            f"{BACKEND_API_URL}/internal/alerts/parent-crisis",
            json={"studentUserId": user_id, "message": message, "subject": subject},
            timeout=3
        )
        print(f"🚨 SENT PARENT ALERT for user {user_id}")
    except Exception as e:
        print(f"❌ Parent alert failed: {e}")

def fetch_teaching_pack_by_chapter_id(chapter_id: str) -> dict:
    try:
        url = f"{BACKEND_API_URL}/internal/teaching-pack/by-chapter/{chapter_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("teachingPack")
        return None
    except Exception as e:
        print(f"⚠️ Error fetching teaching pack by chapterId: {e}")
        return None

def resolve_chapter_for_topic(curriculum_topics: list, chosen_topic_text: str):
    ct = (chosen_topic_text or "").lower().strip()
    for ch in curriculum_topics or []:
        ch_id = ch.get("chapterId") or ch.get("id")  # support either
        ch_title = ch.get("chapterTitle", "")

        for t in ch.get("topics", []) or []:
            t_title = t.get("topicTitle", "") if isinstance(t, dict) else str(t)
            if t_title.lower().strip() == ct:
                return (ch_id, ch_title)

    return (None, None)

async def run_teacher_led_step(websocket: WebSocket, state: Dict, payload: Dict):
    """
    Execute one step of the Teacher-Led Lesson Engine.
    Steps: Teach -> Check -> Practice -> Next Concept
    """
    # Local import to avoid circular dependency risks
    from retrieve_and_respond import judge_mastery, answer_question 

    try:
        lesson = state["lesson"]
        pack = state.get("teacher_pack", {})
        concepts = pack.get("concepts", [])
        
        if not concepts or lesson["step_index"] >= len(concepts):
            # Lesson Complete
            lesson["active"] = False
            await websocket.send_json({
                "message": "🎉 You’ve completed all the core concepts in this chapter! Great job using the Teacher-Led mode.",
                "user_id": "AI_TUTOR"
            })
            state["step"] = "awaiting_topic" # Return to menu
            return

        current_concept = concepts[lesson["step_index"]]
        phase = lesson["phase"]
        
        # Build Teacher Pack Snippet for Context
        tp_snippet = (
            f"Current Chapter: {pack.get('chapterId', '')}\n"
            f"Concept Title: {current_concept.get('title', '')}\n"
            f"Key Content: {current_concept.get('content', '')}\n"
            f"Objectives: {current_concept.get('objectives', '')}\n"
        )

        user_msg = payload.get("message", "") # Student's input

        # 1. TEACH PHASE
        if phase == "teach":
            # Just deliver the content (via answer_question for consistency + personality)
            prompt = f"Teach this concept: {current_concept.get('title')}. Content: {current_concept.get('content')}"
            
            response = answer_question(
                question=prompt,
                history="",
                subject=state.get("subject", "General"),
                student_name=state.get("student_name", "Student"),
                grade_level=state.get("grade_level", "Unknown"),
                teacher_pack_snippet=tp_snippet
            )
            
            # Ask the check question immediately after teaching
            check_q = current_concept.get("check_understanding", "Does that make sense?")
            full_resp = f"{response}\n\n**Check:** {check_q}"
            
            await websocket.send_json({
                "message": full_resp,
                "user_id": "AI_TUTOR"
            })
            
            # Move to CHECK phase to await answer
            lesson["phase"] = "check"
            return

        # 2. CHECK PHASE
        elif phase == "check":
            # Judge the student's answer
            judgment = judge_mastery(
                student_answer=user_msg,
                teach_snippet=tp_snippet,
                context="" 
            )
            
            result = judgment.get("result", "PASS")
            
            if result == "PASS":
                # Mark topic covered
                if lesson.get("topic_title"):
                     state["covered_topics"].append(lesson["topic_title"])

                await websocket.send_json({
                    "message": f"✅ Correct! {judgment.get('reason', 'Good understanding.')}\n\nLet's try a quick practice question to lock it in.",
                    "user_id": "AI_TUTOR"
                })
                # Move to PRACTICE
                lesson["phase"] = "practice"
                
                # Send practice question NOW
                p_q = current_concept.get("practice_question")
                if p_q:
                    await websocket.send_json({
                        "message": f"**Practice:** {p_q}\n(Type your answer)",
                        "user_id": "AI_TUTOR"
                    })
                else:
                    lesson["phase"] = "teach"
                    lesson["step_index"] += 1
                    await websocket.send_json({
                        "message": "Ready for the next concept? (Type 'Ready')",
                        "user_id": "AI_TUTOR"
                    })
                return

            else:
                # FAIL -> Reteach / Hint
                await websocket.send_json({
                    "message": f"🤔 Not quite. {judgment.get('fix', 'Let me explain again.')}\n\n**Try again:** {judgment.get('followup_question', 'What do you think?')}",
                    "user_id": "AI_TUTOR"
                })
                lesson["attempts"] += 1
                # Stay in check phase
                return

        # 3. PRACTICE PHASE
        elif phase == "practice":
            # Simple check for practice
            await websocket.send_json({
                "message": f"Good effort! 🌟\n\nReady for the next concept? (Type 'Ready')",
                "user_id": "AI_TUTOR"
            })
            lesson["step_index"] += 1
            lesson["phase"] = "teach"
            return

    except Exception as e:
        print(f"❌ Lesson Runner Error: {e}")
        traceback.print_exc()
        state["mode"] = "tutoring"
        lesson["active"] = False

def resolve_topic_id_via_api(subject: str, chapter_title: str, topic_title: str) -> dict:
    """
    Resolve legacy string-based topic to standard UUID via Backend API.
    Returns dict { topicId, topicTitle, chapterId } or None.
    """
    try:
        import requests
        url = f"{BACKEND_API_URL}/internal/chapter-topic/resolve-id"
        params = {
            "subject": subject,
            "chapterTitle": chapter_title,
            "topicTitle": topic_title
        }
        # print(f"🔍 Resolving Legacy Topic: {topic_title}...")
        resp = requests.get(url, params=params, timeout=3)
        
        if resp.status_code == 200:
            data = resp.json()
            # print(f"✅ Resolved: {data.get('topicId')}")
            return data
        else:
            print(f"⚠️ Resolution failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"⚠️ Error resolving topic: {e}")
        return None

def fetch_chapter_progress(user_id: str) -> set:
    """
    Fetch completed topic IDs for the student from Node.js Backend.
    Returns a set of topic IDs.
    """
    try:
        import requests
        url = f"{BACKEND_API_URL}/internal/progress/student"
        params = {"userId": user_id}
        # print(f"📊 Fetching student progress...")
        resp = requests.get(url, params=params, timeout=3)
        
        if resp.status_code == 200:
            data = resp.json()
            topics = data.get("topicsCompleted", [])
            # print(f"✅ Progress Loaded: {len(topics)} topics completed")
            return set(topics)
        else:
            print(f"⚠️ Progress fetch failed: {resp.status_code}")
            return set()
    except Exception as e:
        print(f"⚠️ Error fetching progress: {e}")
        return set()

def save_quiz_result(user_id: str, subject: str, quiz_type: str, 
                     total_questions: int, correct_answers: int,
                     chapter_title: str = None, topic_title: str = None, 
                     answers: list = None):
    """
    Save quiz result to backend and return if passed.
    """
    try:
        import requests
        payload = {
            "userId": user_id,
            "subject": subject,
            "quizType": quiz_type,  # 'topic' or 'chapter_final'
            "totalQuestions": total_questions,
            "correctAnswers": correct_answers,
            "chapterTitle": chapter_title,
            "topicTitle": topic_title,
            "answers": answers
        }
        response = requests.post(
            f"{BACKEND_API_URL}/internal/quiz/save-result",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"📝 Quiz saved: {subject} | Score: {data.get('score', 0):.1f}% | Passed: {data.get('passed', False)}")
            return data
        else:
            print(f"⚠️ Quiz save failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Quiz save error: {e}")
        return None

def generate_final_chapter_quiz(subject: str, chapter_title: str, topics: list, 
                                 student_name: str, grade_level: str,
                                 include_past_papers: bool = True) -> str:
    """
    Generate a comprehensive final quiz for a chapter covering all topics.
    Number of questions = number of topics (min 15, max 50).
    Prioritizes past paper questions if available.
    """
    import google.generativeai as genai
    
    # Calculate number of questions based on topics
    num_questions = max(15, min(50, len(topics) * 3))  # ~3 questions per topic
    
    # Create topic list for the prompt
    topics_list = "\n".join([f"- {t}" for t in topics])
    
    # 🚀 NEW: Retrieve Context from RAG
    print(f"🔍 Retrieving context for chapter: {chapter_title}...")
    rag_context = retrieve_and_respond.retrieve_quiz_context(chapter_title, topics, subject)
    
    context_instruction = ""
    if rag_context:
        context_instruction = f"""
START OF CONTEXT FROM TEXTBOOK AND PAST PAPERS:
------------------------------------------------
{rag_context}
------------------------------------------------
END OF CONTEXT

INSTRUCTIONS:
1. Use the above context to create the questions.
2. If "PAST PAPER QUESTIONS" are provided in the context, ADAPT THEM directly into multiple choice format.
3. Ensure the questions cover the provided Topics List.
"""
    else:
        context_instruction = "No specific textbook context found. Use your general knowledge."

    prompt = f"""You are creating a FINAL CHAPTER QUIZ for a {grade_level} student named {student_name}.
Subject: {subject}
Chapter: {chapter_title}

Topics covered:
{topics_list}

{context_instruction}

Generate exactly {num_questions} multiple choice questions.
Return the result strictly as a JSON Object with this structure:
{{
  "title": "Final Quiz: {chapter_title}",
  "questions": [
    {{
      "id": 1,
      "text": "Question text here?",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct_answer": "B",
      "explanation": "Brief explanation why B is correct."
    }}
  ]
}}
IMPORTANT: Return ONLY the JSON. No markdown formatting like ```json ... ```.
"""
    
    # Try Gemini first, fallback to OpenAI
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    generated_text = ""
    
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp", generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(prompt)
            generated_text = response.text
        except Exception as e:
            print(f"⚠️ Gemini quiz error, trying OpenAI: {e}")
    
    # Fallback to OpenAI
    if not generated_text and openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4o-mini", temperature=0.7, model_kwargs={"response_format": {"type": "json_object"}})
            response = llm.invoke(prompt)
            generated_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"⚠️ OpenAI quiz error: {e}")
    
    return generated_text if generated_text else "{}"


def grade_final_quiz(answers_text: str, num_questions: int, quiz_context: str = "") -> tuple:
    """
    Grade quiz answers using AI. Returns (correct_count, total, feedback_text).
    Provides detailed explanation and reasoning for each question.
    """
    import re
    
    prompt = f"""You are grading a quiz. The student's answers are: {answers_text}

Total questions: {num_questions}

{f"Quiz context: {quiz_context}" if quiz_context else ""}

Please grade this quiz and provide a DETAILED REVIEW with the following format:

📊 **QUIZ RESULTS**
━━━━━━━━━━━━━━━━━━━━━
SCORE: X/{num_questions}
PERCENTAGE: XX%
GRADE: [A/B/C/D/F based on percentage]
PASSED: YES/NO (50% or above is passing)

📝 **QUESTION-BY-QUESTION REVIEW**

For EACH question, provide:
**Question 1:** [✅ CORRECT / ❌ INCORRECT]
- Your answer: [letter]
- Correct answer: [letter]
- **Explanation:** [2-3 sentences explaining WHY this is the correct answer and the logic/concept behind it]

**Question 2:** [✅ CORRECT / ❌ INCORRECT]
- Your answer: [letter]
- Correct answer: [letter]
- **Explanation:** [2-3 sentences explaining WHY this is the correct answer]

[Continue for all questions...]

🎯 **KEY TAKEAWAYS**
[2-3 bullet points summarizing the main concepts tested and what the student should focus on]

💪 **ENCOURAGEMENT**
[A motivational message based on their performance]
"""
    
    response_text = ""
    
    # Try Gemini first, fallback to OpenAI
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            print(f"⚠️ Gemini grading error, trying OpenAI: {e}")
    
    # Fallback to OpenAI
    if not response_text and openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4o-mini", temperature=0.3)
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"⚠️ OpenAI grading error: {e}")
            return (0, num_questions, f"Error grading quiz: {e}")
    
    if not response_text:
        return (0, num_questions, "Error: No AI service available to grade quiz")
    
    # Parse the response to extract score
    score_match = re.search(r'SCORE:\s*(\d+)/(\d+)', response_text)
    if score_match:
        correct = int(score_match.group(1))
        total = int(score_match.group(2))
        return (correct, total, response_text)
    else:
        # Fallback - return response as-is
        return (0, num_questions, response_text)


def get_next_curriculum_topics(state: Dict, subject: str, count: int = 3) -> tuple:
    """
    Get next uncovered topics from curriculum.
    Returns: (topics_list, all_covered_flag)
    """
    curriculum = state.get("curriculum_topics", [])
    covered = set(state.get("covered_topics", []))
    
    # Extract all topic titles from curriculum structure
    # Structure: [{chapterTitle, topics: [{topicTitle, topicNumber}]}]
    all_topics = []
    for chapter in curriculum:
        chapter_title = chapter.get("chapterTitle", "")
        for topic in chapter.get("topics", []):
            topic_title = topic.get("topicTitle", "")
            if topic_title:
                all_topics.append(f"{chapter_title}: {topic_title}" if chapter_title else topic_title)
    
    # Filter out covered topics
    uncovered = [t for t in all_topics if t not in covered]
    
    if len(uncovered) == 0:
        # All topics covered - suggest revision
        return ([], True)
    
    # Return next topics (up to count)
    return (uncovered[:count], False)

def get_allowed_topic_ids(state: Dict, quiz_type: str) -> set:
    """
    Determine allowed topic IDs based on quiz type.
    THE QUIZ GATE.
    """
    # 🚨 STRICT LEGACY CHECK
    if state.get("topic_id_mode") == "legacy" and not state.get("current_topic_id"):
        # If we are in legacy mode and failed to resolve an ID, BLOCK ALL QUIIZES.
        # Exception: Final Quiz might work if we strictly use "topics_completed" from DB (which are IDs).
        # But for Contextual/Chip quizzes which rely on "current" context, we must block.
        if quiz_type == "final":
             return state.get("topics_completed", set())
        return set()

    if quiz_type == "contextual":
        # Only the current active topic
        return {state["current_topic_id"]} if state.get("current_topic_id") else set()

    if quiz_type == "chip":
        # Only completed topics
        return state.get("topics_completed", set())

    if quiz_type == "final":
        # Only completed topics (server-side check should enforce *all* chapter topics are here)
        return state.get("topics_completed", set())

    return set()

@app.websocket("/ws/qa_chat/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    
    # 🔍 Extract User ID from Room ID (Subject_UUID_Grade) immediately
    # This ensures timer starts even if they don't send a message
    # Expecting UUID format: 8-4-4-4-12 hex digits
    import re
    import time
    user_id_match = re.search(r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', room_id)
    extracted_user_id = user_id_match.group(1) if user_id_match else None

    if extracted_user_id:
        print(f"⏱️ Starting study timer for: {extracted_user_id} (Connected)", flush=True)
        last_heartbeat[extracted_user_id] = time.time()

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
                "quiz_permission": False,
                "interaction_count": 0,
                "last_quiz_offer": 0,

                # curriculum + progress
                "curriculum_topics": [],
                "covered_topics": [],
                "topics_completed": set(),

                # topic id control
                "current_topic_id": None,
                "current_topic_title": None,
                "topic_id_mode": "legacy",

                # teaching pack
                "teacher_pack": None,

                # safety + alerts
                "mode": "tutoring",
                "had_safety_event": False,
                "last_parent_alert_at": 0,

                # lesson engine (teacher-led)
                "lesson": {
                    "active": False,
                    "phase": "teach",          # teach -> check -> practice
                    "chapter_id": None,
                    "chapter_title": None,
                    "topic_title": None,
                    "step_index": 0,
                    "attempts": 0,
                    "daily_practice_limit": 1,
                    "practice_done_today": 0,
                    "last_practice_date": None
                }
            }
    if room_id not in question_history:
        question_history[room_id] = []

    connections[room_id].append(websocket)

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)

            user_message = payload.get("message", "").strip()
            user_id = payload["user_id"]
            student_name = payload["student_name"]
            grade = payload["grade"]
            subject = payload["subject"]
            curriculum_topics = payload.get("curriculum_topics", [])  # New: get curriculum topics
            msg_type = payload.get("type", "message")

            if msg_type != "init":
                print(f"[{room_id}] {user_id} says: {user_message}")
                print(f"🎓 Student: {student_name} | Grade: {grade} | Subject: {subject}")
            
            # 🚨 0. CRISIS GATE: Detect Self-Harm & Alert
            if msg_type != "init" and detect_self_harm_strong(user_message):
                state = tutor_state[room_id]
                state["mode"] = "safety"
                state["lesson"]["active"] = False

                await websocket.send_json({
                    "message": (
                        f"Hey {student_name} — I’m really glad you told me.\n\n"
                        "I can’t continue the lesson right now because your safety matters most.\n"
                        "Please reach out to a parent/guardian or trusted adult immediately.\n\n"
                        "Are you safe right now? (yes/no)"
                    ),
                    "user_id": "AI_TUTOR",
                    "type": "safety"
                })

                now = time.time()
                if now - state.get("last_parent_alert_at", 0) >= ALERT_THROTTLE_SECONDS:
                    state["last_parent_alert_at"] = now
                    trigger_parent_alert_via_node(user_id, user_message, subject)

                continue
            
            # Send study heartbeat to track time for streak
            send_study_heartbeat(user_id)

            # Defensive: Re-initialize state if it was cleared by a racing disconnect
            if room_id not in connections:
                connections[room_id] = []
            if room_id not in chat_logs:
                chat_logs[room_id] = []  # ✅ CRITICAL FIX: Ensure list exists
            if room_id not in welcome_sent:
                welcome_sent[room_id] = False
            if room_id not in tutor_state:
                tutor_state[room_id] = {
                    "step": "awaiting_topic",
                    "topic": None,
                    "level": None,
                    "prior_knowledge": None,
                    "quiz_permission": False,
                    "interaction_count": 0,
                    "last_quiz_offer": 0,
                    "curriculum_topics": [],
                    "covered_topics": [],
                    "teacher_pack": None,  # Store teaching pack here
                    "current_topic_id": None,     # ✅ NEW
                    "current_topic_title": None,  # ✅ NEW
                    "topics_completed": set()     # ✅ NEW
                }
            if room_id not in question_history:
                question_history[room_id] = []

            state = tutor_state[room_id]
            
            # Update curriculum topics if provided (first message usually has them)
            if curriculum_topics and len(curriculum_topics) > 0:
                state["curriculum_topics"] = curriculum_topics

            if not welcome_sent.get(room_id, False):
                welcome_sent[room_id] = True

                # Load chat history for this user and subject
                history_entries = get_user_chat_history(user_id, subject)
                
                if len(history_entries) > 0:
                    # Load previous chat history into current session
                    if room_id not in chat_logs: chat_logs[room_id] = [] # Double check
                    
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
                
                # 🟢 START PERIODIC HEARTBEAT LOOP
                asyncio.create_task(study_heartbeat_loop(user_id, room_id))
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
            if state["step"] == "awaiting_topic" and msg_type != "init":
                state["topic"] = user_message
                
                # 🚀 FETCH TEACHING PACK HERE (if not already loaded or if simplified flow)
                # Ideally we want it at room init if we knew the chapter. 
                # But topic selection is often where we narrow down.
                # Actually, the user selects a "Chapter" in UI, but writes a "Topic". 
                # Let's try to fetch if we have curriculum info.
                
                # Try to find which chapter this topic belongs to from curriculum_topics
                chapter_for_pack = subject # Default
                
                # ✅ NEW: Reset current topic ID/Title
                state["current_topic_title"] = user_message
                state["current_topic_id"] = None
                state["topic_id_mode"] = "legacy" # Assume legacy until proven otherwise

                # Resolve Chapter ID & Topic Model
                chapter_for_pack = subject # fallback
                
                if state.get("curriculum_topics"):
                    # user_message is the topic title
                    for ch in state["curriculum_topics"]:
                         for t in ch.get("topics", []):
                            # Handle both object (new) and string (old) formats
                            t_title = t.get("topicTitle", "") if isinstance(t, dict) else t
                            if t_title.lower() == user_message.lower():
                                chapter_for_pack = ch.get("chapterTitle", subject)
                                if isinstance(t, dict) and "topicId" in t:
                                    state["current_topic_id"] = t["topicId"] # ✅ Capture ID
                                    state["topic_id_mode"] = "id"           # ✅ Mode: ID
                                break
                
                # 🚑 SELF-HEALING: If Legacy Mode (No ID found), attempt backend resolution
                if state["topic_id_mode"] == "legacy" and not state["current_topic_id"]:
                    print(f"🩹 Attempting self-healing for topic: {user_message}")
                    loop = asyncio.get_event_loop()
                    resolved = await loop.run_in_executor(
                        None, 
                        resolve_topic_id_via_api, 
                        subject, 
                        chapter_for_pack, 
                        user_message
                    )
                    if resolved and resolved.get("topicId"):
                        state["current_topic_id"] = resolved["topicId"]
                        state["topic_id_mode"] = "id"
                        print(f"✅ Self-healed session! Switched to ID mode: {state['current_topic_id']}")
                    else:
                        print("⚠️ Self-healing failed. Quiz capabilities limited.")

                # ✅ NEW: Fetch Teaching Pack by Chapter ID (if available from curriculum or resolve)
                # Resolve chapter info for pack fetching
                chap_id, chap_title = resolve_chapter_for_topic(state.get("curriculum_topics", []), user_message)
                
                if chap_id:
                     print(f"📦 Found Chapter ID: {chap_id}, fetching pack...")
                     loop = asyncio.get_event_loop()
                     pack = await loop.run_in_executor(None, fetch_teaching_pack_by_chapter_id, chap_id)
                     if pack:
                         state["teacher_pack"] = pack
                         state["lesson"]["chapter_id"] = chap_id
                         state["lesson"]["chapter_title"] = chap_title or subject
                         print("✅ Teaching Pack Loaded via Chapter ID!")
                     else:
                         print("⚠️ Failed to load pack via Chapter ID, falling back to legacy flow if needed.")
                else:
                    print("⚠️ Could not resolve Chapter ID from provided curriculum.")

                # ✅ Sync DB Progress
                loop = asyncio.get_event_loop()
                completed_set = await loop.run_in_executor(None, fetch_chapter_progress, user_id)
                state["topics_completed"] = completed_set
                
                # 👋 START LESSON IMMEDIATELY (B5)
                if state.get("teacher_pack"):
                    state["lesson"]["active"] = True
                    state["lesson"]["phase"] = "teach"
                    state["lesson"]["topic_title"] = user_message
                    state["lesson"]["step_index"] = 0
                    state["lesson"]["attempts"] = 0
                    state["step"] = "tutoring" # Skip awaiting_prior_knowledge

                    await websocket.send_json({
                        "message": f"Perfect. I’ll teach **{state['lesson']['chapter_title']}** step-by-step (teach → check → practice). Let’s begin.",
                        "user_id": "AI_TUTOR"
                    })
                    # ✅ run the teacher-led engine instead of free chat
                    await run_teacher_led_step(websocket, state, payload)
                    continue
                else:
                    # Fallback to old flow if no pack
                    state["step"] = "awaiting_prior_knowledge"
                    await websocket.send_json({
                        "message": f"Great! We'll focus on **{state['topic']}**. How familiar are you with this topic? (Beginner, Intermediate, Advanced)",
                        "user_id": "AI_TUTOR"
                    })
                    continue
                # print(f"🔄 State Synced. Current Topic ID: {state['current_topic_id']} | Completed: {len(state['topics_completed'])}")
                
                if not state.get("teacher_pack"):
                     # Run in background to not block
                     loop = asyncio.get_event_loop()
                     pack = await loop.run_in_executor(None, fetch_teaching_pack, subject, chapter_for_pack)
                     if pack:
                         state["teacher_pack"] = pack

                state["topic"] = user_message
                # Infer level from grade (already provided in payload)
                # Grade format: "Grade 10", "Grade 12", etc.
                if "grade" in grade.lower() or any(g in grade.lower() for g in ["9", "10", "11", "12"]):
                    state["level"] = "high school student"
                elif any(g in grade.lower() for g in ["college", "university", "undergrad"]):
                    state["level"] = "college student"
                else:
                    state["level"] = grade  # Use grade as-is if format is different
                
                # ✅ Update chapter progress when topic is selected
                update_chapter_progress(
                    user_id=user_id,
                    subject=subject,
                    topic_title=user_message,
                    chapter_title=subject
                )
                
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
                    "message": "Awesome! Getting the quiz ready...",
                    "user_id": "AI_TUTOR"
                })
                # Trigger quiz immediately if they agreed
                user_message = "quiz" 
                # Fall through to quiz handler below
            
            if user_message.lower() in ["no", "not now"] and state["step"] == "tutoring":
                 state["quiz_permission"] = False
                 await websocket.send_json({
                    "message": "No problem! Let's keep exploring. What's on your mind?",
                    "user_id": "AI_TUTOR"
                })
                 continue

            if user_message.lower() in ["yes quiz", "start quiz", "quiz", "let's do quiz"] and state["quiz_permission"]:
                history = "\n".join(chat_logs[room_id][-6:])
                
                # 🧠 Compute allowed topic IDs using the Gate
                is_contextual = True # Default to contextual for immediate "quiz me" flow? 
                # If they say "quiz", they usually mean "quiz me on what we just talked about".
                # If they say "general quiz", maybe not. Let's assume contextual if current topic is set.
                quiz_mode = "contextual" if state.get("current_topic_id") else "final" # Fallback to broad if no topic
                
                allowed_ids = get_allowed_topic_ids(state, quiz_mode)
                
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
                        mode="quiz",
                        current_topic=state["topic"],
                        # ✅ NEW: Use centralized Gate
                        allowed_topic_ids=list(allowed_ids)
                    )
                )
                await websocket.send_json({"message": quiz_response, "user_id": "AI_TUTOR"})
                continue

            # === FINAL CHAPTER QUIZ HANDLER ===
            # Triggers: "final quiz", "chapter quiz", "final quiz for [chapter]"
            final_quiz_keywords = ["final quiz", "chapter quiz", "chapter final quiz", "take final quiz"]
            is_final_quiz_request = any(kw in user_message.lower() for kw in final_quiz_keywords)
            
            if is_final_quiz_request:
                # Extract chapter from message or use current chapter from curriculum
                chapter_to_quiz = None
                topics_for_quiz = []
                
                # Try to find chapter in curriculum_topics
                curriculum = state.get("curriculum_topics", [])
                
                # Check if a specific chapter is mentioned
                for chapter in curriculum:
                    chapter_title = chapter.get("chapterTitle", "")
                    if chapter_title.lower() in user_message.lower():
                        chapter_to_quiz = chapter_title
                        topics_for_quiz = [t.get("topicTitle", "") for t in chapter.get("topics", [])]
                        break
                
                # If no specific chapter mentioned, use the current topic's chapter or first chapter
                if not chapter_to_quiz and curriculum:
                    # Try to find which chapter current topic belongs to
                    current_topic = state.get("topic", "")
                    for chapter in curriculum:
                        chapter_title = chapter.get("chapterTitle", "")
                        for topic in chapter.get("topics", []):
                            t_title = topic.get("topicTitle", "") if isinstance(topic, dict) else topic
                            if current_topic and t_title in current_topic:
                                chapter_to_quiz = chapter_title
                                break
                        if chapter_to_quiz:
                            break
                    
                    # Still nothing? Use first chapter
                    if not chapter_to_quiz and len(curriculum) > 0:
                        chapter_to_quiz = curriculum[0].get("chapterTitle", subject)
                
                if not chapter_to_quiz:
                    chapter_to_quiz = subject

                # 🚀 SCOPE CHECK for Final Quiz
                # Verify all topics in this chapter are completed
                chapter_obj = next((ch for ch in curriculum if ch.get("chapterTitle") == chapter_to_quiz), None)
                all_chapter_topic_ids = []
                topics_for_quiz = []
                
                if chapter_obj:
                    for t in chapter_obj.get("topics", []):
                        if isinstance(t, dict):
                            all_chapter_topic_ids.append(t.get("topicId"))
                            topics_for_quiz.append(t.get("topicTitle"))
                        else:
                            # Handling legacy format (string only) - assume no ID
                            topics_for_quiz.append(t)
                
                completed_ids = state.get("topics_completed", set())
                # Filter out None IDs if any
                valid_ids = [tid for tid in all_chapter_topic_ids if tid]
                
                # Check completion (strict mode)
                if valid_ids and not all(tid in completed_ids for tid in valid_ids):
                    await websocket.send_json({
                        "message": f"🚫 **Hold on!** You haven't finished all the topics in **{chapter_to_quiz}** yet.\n\nPlease complete all topics to unlock the Final Chapter Quiz!",
                        "user_id": "AI_TUTOR"
                    })
                    continue

                if not topics_for_quiz:
                    topics_for_quiz = [state.get("topic", "General")]
                
                # Initialize Quiz State
                state["final_quiz_active"] = True
                state["final_quiz_chapter"] = chapter_to_quiz
                state["final_quiz_topics"] = topics_for_quiz
                state["final_quiz_num_questions"] = max(15, min(50, len(topics_for_quiz) * 3))
                state["quiz_data"] = None 
                
                await websocket.send_json({
                    "message": f"📚 Generating Final Chapter Quiz for **{chapter_to_quiz}**...\n\nPlease wait while I analyze the textbook content and past papers... ⏳",
                    "user_id": "AI_TUTOR"
                })
                
                # Generate the final quiz JSON
                loop = asyncio.get_event_loop()
                quiz_json_str = await loop.run_in_executor(
                    None,
                    partial(
                        generate_final_chapter_quiz,
                        subject=subject,
                        chapter_title=chapter_to_quiz,
                        topics=topics_for_quiz,
                        student_name=student_name,
                        grade_level=grade,
                        include_past_papers=True
                    )
                )

                try:
                    # Extract JSON if it's wrapped in markdown code blocks
                    if "```json" in quiz_json_str:
                        quiz_json_str = quiz_json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in quiz_json_str:
                        quiz_json_str = quiz_json_str.split("```")[1].split("```")[0].strip()
                    
                    quiz_data = json.loads(quiz_json_str)
                    state["quiz_data"] = quiz_data
                    state["quiz_questions"] = quiz_data.get("questions", [])
                    
                    print(f"🧩 Quiz Generated. Questions Count: {len(state['quiz_questions'])}", flush=True)

                    if not state["quiz_questions"]:
                        raise ValueError("No questions found in generated JSON")
                    
                    # 🚀 FORMAT AS SINGLE STRING FOR FRONTEND CAROUSEL
                    # The frontend looks for "1. Question ... A) ..." pattern
                    # We ADD metadata for the frontend to parse: (correct) marker and [Explanation]
                    full_quiz_text = f"**FINAL QUIZ: {chapter_to_quiz}**\n\n"
                    
                    for i, q in enumerate(state["quiz_questions"]):
                        full_quiz_text += f"{i+1}. {q['text']}\n"
                        correct_char = q.get('correct_answer', 'A').strip().upper()[0]
                        
                        for opt in q.get('options', []):
                            # Check if this option is the correct one
                            # opt usually looks like "A) Text"
                            is_correct = opt.strip().upper().startswith(f"{correct_char})") or \
                                         opt.strip().upper().startswith(f"{correct_char}.")
                            
                            clean_opt = opt.strip()
                            if is_correct:
                                clean_opt += " (correct)"
                            
                            full_quiz_text += f"{clean_opt}\n"
                        
                        # Add explanation block (Frontend will hide this but use it for feedback)
                        expl = q.get('explanation', '')
                        if expl:
                            full_quiz_text += f"[Explanation: {expl}]\n"
                            
                        full_quiz_text += "\n" # Spacing
                    
                    # Send the FULL QUIIZ at once to trigger the Carousel UI
                    await websocket.send_json({
                        "message": full_quiz_text,
                        "user_id": "AI_TUTOR"
                    })

                except Exception as e:
                    print(f"❌ Error parsing quiz JSON: {e}", flush=True)
                    state["final_quiz_active"] = False # Abort
                    await websocket.send_json({
                        "message": "⚠️ Sorry, I had trouble generating the quiz correctly. Please try again!",
                        "user_id": "AI_TUTOR"
                    })
                continue
            
            # === HANDLE QUIZ SUBMISSION (BATCH) ===
            # The frontend sends "1-a, 2-b, 3-c..." when "Submit for review" is clicked
            import re
            batch_answer_pattern = re.compile(r'^\s*(\d+\s*[-]\s*[A-Da-d]\s*[,\s]*)+\s*$')
            
            if state.get("final_quiz_active") and batch_answer_pattern.match(user_message):
                print(f"📝 Quiz Submission Received: {user_message}", flush=True)
                
                chapter_title = state.get("final_quiz_chapter", subject)
                
                await websocket.send_json({
                    "message": "📝 Grading your quiz... please wait!",
                    "user_id": "AI_TUTOR"
                })
                
                # Parse the user's batch string "1-a, 2-b"
                user_answers_map = {}
                parts = user_message.split(',')
                for p in parts:
                    if '-' in p:
                        q_num, ans_char = p.split('-')
                        user_answers_map[int(q_num.strip())] = ans_char.strip().upper()

                # Grade against state['quiz_questions']
                correct_count = 0
                questions = state.get("quiz_questions", [])
                total = len(questions)
                
                # (Review details not strictly needed if we do realtime, but good for summary)
                
                for i, q in enumerate(questions):
                    q_id = i + 1
                    user_ans = user_answers_map.get(q_id, "?")
                    correct_ans = q.get("correct_answer", "A").strip().upper()[0]
                    if user_ans == correct_ans:
                        correct_count += 1

                percentage = (correct_count / total) * 100 if total > 0 else 0
                passed = percentage >= 50
                
                # Save result
                quiz_result = save_quiz_result(
                    user_id=user_id,
                    subject=subject,
                    quiz_type="chapter_final",
                    total_questions=total,
                    correct_answers=correct_count,
                    chapter_title=chapter_title,
                    answers=[user_message]
                )
                
                state["final_quiz_active"] = False
                
                # --- NEW FINAL RESULT FORMAT ---
                result_msg = f"📊 **QUIZ RESULTS: {chapter_title}**\n\n"
                result_msg += f"🏆 **Score:** {correct_count} / {total} ({percentage:.1f}%)\n"
                result_msg += f"✅ **Status:** {'PASSED' if passed else 'TRY AGAIN'}\n"
                
                if passed:
                    result_msg += f"**Chapter {chapter_title} is now complete**\n"
                    
                    # Badge Logic (Aligned with Backend: Einstein, Newton, Curie, Galileo)
                    # 100% -> Einstein
                    # 85-99% -> Newton
                    # 70-84% -> Curie
                    # 65-69% -> Galileo
                    
                    if percentage == 100:
                        result_msg += f"**You have earned {subject} Einstein Badge!** 🧠🏆\n"
                    elif percentage >= 85:
                        result_msg += f"**You have earned {subject} Newton Badge!** 🍎\n"
                    elif percentage >= 70:
                        result_msg += f"**You have earned {subject} Curie Badge!** 🧪\n"
                    elif percentage >= 65:
                        result_msg += f"**You have earned {subject} Galileo Badge!** 🔭\n"
                    
                    result_msg += "\n🎉 Outstanding work! You've mastered this chapter."
                else:
                    result_msg += "\nKeep practicing! You're getting there."

                await websocket.send_json({"message": result_msg, "user_id": "AI_TUTOR"})
                
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

            # Check for repeated SAME questions (only in tutoring mode)
            if state["step"] == "tutoring":
                previous_questions = question_history.get(room_id, [])
                if len(previous_questions) >= 3:  # Need at least 3 previous questions to check
                    is_similar, similar_count, most_similar_q = check_question_similarity(
                        user_message, 
                        previous_questions,
                        similarity_threshold=0.95  # Very high threshold = essentially the same question
                    )
                    
                    # Only suggest moving on if the SAME question is asked more than 3 times
                    if is_similar and similar_count > 3:  
                        # Get next topics from curriculum (not AI-generated)
                        next_topics, all_covered = get_next_curriculum_topics(state, subject, count=3)
                        
                        # Mark current topic as covered
                        if state.get("topic"):
                            if "covered_topics" not in state:
                                state["covered_topics"] = []
                            if state["topic"] not in state["covered_topics"]:
                                state["covered_topics"].append(state["topic"])
                        
                        if all_covered:
                            # All topics done - offer revision/quiz
                            await websocket.send_json({
                                "message": f"🎉 Amazing! You've explored all the topics in **{subject}**! Let's reinforce your learning with a quick revision quiz!",
                                "user_id": "AI_TUTOR"
                            })
                            state["quiz_permission"] = True
                        else:
                            await websocket.send_json({
                                "message": f"🔄 I notice you've asked this exact question multiple times. Let's try a fresh approach or explore a new topic in **{subject}**!",
                                "user_id": "AI_TUTOR"
                            })
                            await websocket.send_json({
                                "message": f"📘 Here are the next topics from your curriculum:",
                                "topics": next_topics,
                                "type": "topic_suggestions",
                                "user_id": "AI_TUTOR"
                            })
                        # Clear question history to start fresh with new topic
                        question_history[room_id] = []
                        continue
                    elif is_similar and similar_count >= 2:
                        # 2-3 similar questions - still answer but log it
                        print(f"⚠️ Same question detected ({similar_count} times). Most similar: {most_similar_q}")

            # Track the question
            question_history[room_id].append(user_message)
            # Keep only last 20 questions to avoid memory issues
            question_history[room_id] = question_history[room_id][-20:]

            # Regular tutoring flow
            history = "\n".join(chat_logs[room_id][-6:])

            # Handle CHIP_CLICK (Structured Interaction)
            if msg_type == "CHIP_CLICK":
                # Use handle_chip_click from retrieve_and_respond
                teacher_pack = state.get("teacher_pack", {})
                
                # Generate AI response based on chip intent
                ai_response_text = handle_chip_click(payload, teacher_pack, state.get("topics_completed", set()))
                
                # Send immediate response
                chat_logs[room_id].append(f"User (Chip): {payload.get('label')}")
                chat_logs[room_id].append(f"AI: {ai_response_text}")
                chat_logs[room_id] = chat_logs[room_id][-10:]
                
                await websocket.send_json({"message": ai_response_text, "user_id": "AI_TUTOR"})
                
                # Generate follow-up suggestions (New Chips)
                try:
                    loop = asyncio.get_event_loop()
                    suggestions = await loop.run_in_executor(
                        None,
                        partial(
                            generate_prompt_suggestions,
                            question=payload.get("label", "Next step"), # Use label as context
                            response=ai_response_text,
                            subject=subject,
                            student_name=student_name,
                            grade_level=grade,
                            history=history,
                            teacher_pack=teacher_pack # ✅ Pass pack
                        )
                    )
                    
                    if suggestions and len(suggestions) > 0:
                        await websocket.send_json({
                            "message": "💡 Next:",
                            "suggestions": suggestions,
                            "type": "prompt_suggestions",
                            "user_id": "AI_TUTOR"
                        })
                except Exception as e:
                    print(f"⚠️ Error generating chip follow-ups: {e}")
                
                continue # Skip standard flow
            
            # Extract country and board from payload (default to Unknown if missing)
            # 1. Try payload
            country = payload.get("country")
            board = payload.get("board") or payload.get("curriculum")
            
            # 2. If present, update state
            if country:
                state["country"] = country
            if board:
                state["board"] = board
                
            # 3. If missing in payload, fallback to state
            if not country:
                country = state.get("country", "Unknown")
            if not board:
                board = state.get("board", "Unknown")

            # Extract allowed subtopics from curriculum
            
            # Extract allowed subtopics from curriculum
            allowed_subtopics = []
            if "curriculum_topics" in payload:
                for ch in payload.get("curriculum_topics", []):
                    # Flatten topics
                    for t in ch.get("topics", []):
                        if isinstance(t, dict):
                            allowed_subtopics.append(t.get("topicTitle", ""))
                        elif isinstance(t, str):
                            allowed_subtopics.append(t)
            
            # Fallback for current_topic: if None, use Subject
            active_topic_context = state.get("topic")
            if not active_topic_context:
                active_topic_context = subject

            # 🛠️ LESSON ENGINE HOOK (B6)
            # Before calling generic answer_question, check if we are in a structured lesson
            if state.get("lesson", {}).get("active", False):
                 # Pass through to teacher-led runner
                 # We assume `user_message` is the student's response to the previous prompt (teach acknowledgement, or check answer)
                 await run_teacher_led_step(websocket, state, payload)
                 continue

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
                    country=country,
                    board=board,
                    mode="tutoring",
                    current_topic=active_topic_context,
                    allowed_subtopics=allowed_subtopics 
                )
            )
            
            # Increment interaction count
            state["interaction_count"] += 1
            
            # Update chapter progress in dashboard (first interaction with a topic)
            if state["interaction_count"] == 1 and state.get("topic"):
                update_chapter_progress(
                    user_id=user_id,
                    subject=subject,
                    topic_title=state["topic"],
                    chapter_title=subject
                )

            response_text = str(ai_response.content) if hasattr(ai_response, "content") else str(ai_response)
            chat_logs[room_id].append(f"User: {user_message}")
            chat_logs[room_id].append(f"AI: {response_text}")
            chat_logs[room_id] = chat_logs[room_id][-10:]

            # 🚀 SEND THE MAIN RESPONSE FIRST (before generating suggestions)
            await websocket.send_json({"message": response_text, "user_id": "AI_TUTOR"})
            
            # Periodically offer a quiz (e.g., every 4 interactions)
            if state["interaction_count"] - state["last_quiz_offer"] >= 4:
                state["last_quiz_offer"] = state["interaction_count"]
                state["quiz_permission"] = False # Reset permission
                await asyncio.sleep(0.5)
                await websocket.send_json({
                    "message": f"We've covered some cool stuff about {state['topic']}! 🧠 Want to try a quick quiz to test your skills?",
                    "user_id": "AI_TUTOR"
                })
                state["quiz_permission"] = False # Logic check: require specific "yes" for quiz

            # 💡 Generate suggestions AFTER sending main response (non-blocking for user)
            # Only for regular conversations, not quizzes
            if "Choose A, B, C, or D" not in response_text:
                try:
                    loop = asyncio.get_event_loop()
                    suggestions = await loop.run_in_executor(
                        None,
                        partial(
                            generate_prompt_suggestions,
                            question=user_message,
                            response=response_text,
                            subject=subject,
                            student_name=student_name,
                            grade_level=grade,
                            student_name=student_name,
                            grade_level=grade,
                            history=history,
                            teacher_pack=state.get("teacher_pack") # ✅ Pass pack
                        )
                    )
                    
                    if suggestions and len(suggestions) > 0:
                        await websocket.send_json({
                            "message": "💡 Here are some follow-up questions you might ask:",
                            "suggestions": suggestions,
                            "type": "prompt_suggestions",
                            "user_id": "AI_TUTOR"
                        })
                except Exception as e:
                    print(f"⚠️ Error generating suggestions: {e}")

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

        # 🛑 STOP TIMER on disconnect
        if extracted_user_id and extracted_user_id in last_heartbeat:
            del last_heartbeat[extracted_user_id]
            print(f"🛑 Stopped study timer for: {extracted_user_id} (Disconnected)", flush=True)

# ============================================================================
# FILE UPLOAD ENDPOINT FOR RAG EMBEDDING
# ============================================================================

@app.post("/api/upload")
async def upload_file_for_embedding(
    file: UploadFile = File(...),
    subject: str = Form(...),
    grade: Optional[str] = Form(None),
    curriculum: Optional[str] = Form(None),
    board: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
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
        filename = os.path.basename(file.filename)
        save_path = f"uploads/{filename}"
        
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
        if board:
            metadata["board"] = board
        if country:
            metadata["country"] = country
        
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
        print(f"❌ Upload Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.post("/api/generate-teaching-pack")
async def create_teaching_pack(
    file: UploadFile = File(...)
):
    """
    Generate a JSON Teaching Pack from an uploaded PDF.
    """
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        filename = f"temp_tp_{file.filename}"
        save_path = f"uploads/{filename}"
        
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
            
        print(f"🧠 Generating Teaching Pack for: {filename}...")
        pack = generate_teaching_pack(save_path)
        
        # Cleanup temp file
        # os.remove(save_path) 
        
        return {"status": "success", "teaching_pack": pack}
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
