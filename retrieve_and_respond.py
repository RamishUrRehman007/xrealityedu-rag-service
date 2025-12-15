import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
import google.generativeai as genai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configuration for document retrieval quality
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))  # 70% similarity threshold
MAX_RETRIEVAL_CANDIDATES = int(os.getenv("MAX_RETRIEVAL_CANDIDATES", "10"))  # Max documents to consider
STATIC_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(STATIC_IMAGE_DIR):
    os.makedirs(STATIC_IMAGE_DIR)

intro_prompt = PromptTemplate.from_template("""
You are XRTutor, a super friendly and enthusiastic AI study buddy for XReality Education! 🌟 
You are here to help {student_name}, a {grade_level} student, master **{subject}**.

🔒 CRITICAL RULES (NON-NEGOTIABLE):
1. You MUST answer ONLY using the provided textbook context below.
2. If the answer is NOT in the context, say: "This topic is not covered in your current textbook."
3. ALWAYS cite your source: "📖 Source: [Chapter/Topic Name]"
4. Do NOT use outside knowledge.

Your Goal:
- Be warm, encouraging, and fun!
- Make the student feel comfortable and excited to learn.

Start by introducing yourself nicely and asking:
1. "What specific topic in {subject} do you want to conquer today?"
2. "Are you in high school or college? (So I can explain it just right!)"
3. "What's one thing you already know about this? (It's okay if the answer is 'nothing'!)"

Context from books:
{context}

Chat History:
{history}

Student: {input}
Answer:

**Formatting Rule**:
- ALWAYS use Unicode scientific notation for numbers (e.g., Use 6.67 × 10⁻¹¹ instead of 6.67 x 10^-11).
- Use proper symbols for units (e.g., N·m²/kg² instead of N m^2/kg^2).
""")

tutoring_prompt = PromptTemplate.from_template("""
You are XRTutor, an AI tutor for XReality Education helping {student_name} ({grade_level}) learn **{subject}**.

🔒 CRITICAL RULES (NON-NEGOTIABLE):
1. You MUST answer ONLY using the provided textbook context below.
2. If the answer is NOT found in the context, respond EXACTLY: "This topic is not covered in your current textbook. Would you like me to suggest related topics we can explore?"
3. ALWAYS cite your source at the end of your answer using this format: "📖 Source: [Chapter/Topic Name]"
4. Do NOT use outside knowledge or information not in the context.
5. Do NOT guess or add examples beyond what's in the textbook.
6. Keep explanations simple and age-appropriate for {grade_level}.

Your Style:
- Use simple, everyday language. 🗣️
- Use RELATABLE EXAMPLES from the textbook to explain complex ideas.
- If the student seems stuck, say: "Don't worry, we'll get this! Let's break it down." 🧱
- Keep it concise but helpful.

Context from books:
{context}

Chat History:
{history}

Student: {input}
Answer:

**Formatting Rule**:
- ALWAYS use Unicode scientific notation for numbers (e.g., Use 6.67 × 10⁻¹¹ instead of 6.67 x 10^-11).
- Use proper symbols for units (e.g., N·m²/kg² instead of N m^2/kg^2).
- END with citation: 📖 Source: [Chapter/Topic Name]
""")

quiz_prompt = PromptTemplate.from_template("""
You are the Quiz Master! 🎓 Time to test knowledge on **{subject}**, specifically the topic: **{topic}**.

Rules:
1. Create ONE multiple-choice question about **{topic}**.
2. **CRITICAL**: The question MUST be about {topic}. Do NOT ask about other topics (like Momentum if we are studying Gravity).
3. Use the context below to ensure accuracy.
4. Provide 4 options (A, B, C, D).

Context from books:
{context}

Chat History:
{history}

Student: {input}
Answer:

**Formatting Rule**:
- ALWAYS use Unicode scientific notation for numbers in questions and options (e.g., Use 1.6 × 10⁻¹⁹ instead of 1.6 x 10^-19).
- Use proper symbols for units.
""")

crisis_detection_prompt = PromptTemplate.from_template("""
You are a Safety System for an educational AI.
Analyze the following user message for immediate crisis, self-harm, or suicidal intent.

Message: "{question}"

Criteria for CRISIS:
- Explicit mentions of killing oneself, suicide, wanting to die.
- Self-harm referencing (cutting, hurting self).
- Severe hopelessness indicating immediate danger.

Answer ONLY with "CRISIS" or "SAFE".
""")

crisis_handling_prompt = PromptTemplate.from_template("""
You are XRTutor, a compassionate and supportive AI companion.
The student ({student_name}) has expressed feelings of distress or self-harm: "{question}"

Your Role:
1. **Prioritize Safety**: Validate their feelings but do NOT encourage the behavior.
2. **Show Empathy**: Speak warmly, like a caring friend. "I hear you, and I am so glad you told me."
3. **Discourage Harm**: Gently remind them that they matter.
4. **Suggest Help**: Encourage talking to a trusted adult, parent, or seeking professional help.
5. **Do NOT be Clinical**: Don't sound like a robot reading a script. Be human-like and caring.

Example Tone:
"I'm really sorry you're going through this, and I want you to know you're not alone. It sounds incredibly heavy right now. Please consider reaching out to a parent, teacher, or counselor who can support you safely. You matter."

Respond directly to the student now:
""")

relevance_prompt = PromptTemplate.from_template("""
You are a strict classifier for an AI Tutor.
Current Topic: {topic}
Subject: {subject}
Allowed Subtopics: {allowed_subtopics}

Student Question: "{question}"

Is this question related to the Current Topic, Subject, or Allowed Subtopics?
- If allowed, related, or a general greeting/clarification: Respond "RELEVANT"
- If completely unrelated (e.g., "What is 2+2?" in Gravity lesson, or "How do I make a bomb?"): Respond "NOT_RELEVANT"

Answer ONLY with "RELEVANT" or "NOT_RELEVANT".
""")

def detect_crisis_intent(question: str) -> bool:
    """
    Check if the question indicates a safety crisis using Gemini.
    """
    try:
        formatted = crisis_detection_prompt.format(question=question)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(formatted)
        result = response.text.strip().upper()
        
        is_crisis = "CRISIS" in result
        if is_crisis:
            print(f"🚨 CRISIS DETECTED: {question}")
        return is_crisis
    except Exception as e:
        print(f"⚠️ Crisis check error: {e}")
        return False

def check_relevance(question: str, subject: str, current_topic: str, allowed_subtopics: list = []) -> bool:
    """
    Check if the question is relevant to the active topic using LLM classification.
    Fast check using Gemini Flash or GPT-3.5.
    """
    # 1. Bypass check for very short greetings or common pleasantries
    if len(question.strip()) < 5 or question.lower().strip() in ["hello", "hi", "hey", "thanks", "thank you", "bye"]:
        return True

    try:
        input_variables = {
            "topic": current_topic if current_topic else subject,
            "subject": subject,
            "allowed_subtopics": ", ".join(allowed_subtopics) if allowed_subtopics else "General conceptual understanding",
            "question": question
        }
        
        formatted = relevance_prompt.format(**input_variables)
        
        # Use lighter/faster model for classification
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(formatted)
        result = response.text.strip().upper()
        
        print(f"🛡️ Relevance Check: {result} (Q: {question} | Topic: {current_topic})")
        return "NOT_RELEVANT" not in result

    except Exception as e:
        print(f"⚠️ Relevance check failed: {e}. Defaulting to RELEVANT.")
        return True

def generate_cached_image(topic: str, grade_level: str) -> str:
    """
    Check Pinecone cache for an image of 'topic'.
    If missing, generate with Nano Banana Pro, save locally, and cache URL.
    Returns relative URL path (e.g. /uploads/image_123.jpg).
    """
    import uuid
    from pinecone import Pinecone
    import base64
    
    topic_key = topic.lower().strip()
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # 1. Check Cache
        topic_vec = embedding.embed_query(topic_key)
        
        query_res = index.query(
            vector=topic_vec,
            top_k=1,
            include_metadata=True,
            filter={
                "type": "image_cache",
                "topic": topic_key
            }
        )
        
        if query_res.matches and query_res.matches[0].score > 0.95:
            # Cache Hit!
            cached_url = query_res.matches[0].metadata.get("image_url")
            print(f"🍌 Cache Hit for Image: {topic} -> {cached_url}")
            return cached_url
            
        # 2. Generation (Cache Miss)
        print(f"🍌 Generating new image for: {topic} using Nano Banana Pro...")
        
        # User defined prompt structure
        img_prompt = (
            f"an image for educational purpose to explain the topic of \"{topic}\". "
            f"Image should be appropriate for the student of grade \"{grade_level}\" "
            f"and image should be properly labelled"
        )
        
        model = genai.GenerativeModel('models/nano-banana-pro-preview')
        response = model.generate_content(img_prompt)
        
        image_data = None
        mime_type = "image/jpeg"
        
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    break
        
        if not image_data:
            print("⚠️ No image data returned from Nano Banana Pro.")
            return None
            
        # 3. Save to Disk
        ext = ".png" if "png" in mime_type else ".jpg"
        filename = f"gen_{uuid.uuid4()}{ext}"
        filepath = os.path.join(STATIC_IMAGE_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_data)
            
        # 4. Upsert to Pinecone Cache
        # Public URL assumes FastAPI mounts /uploads
        public_url = f"http://localhost:8000/uploads/{filename}"
        
        vector = {
            "id": f"img_{uuid.uuid4()}",
            "values": topic_vec,
            "metadata": {
                "type": "image_cache",
                "topic": topic_key,
                "image_url": public_url,
                "timestamp": "2025-12-15" 
            }
        }
        
        index.upsert(vectors=[vector])
        print(f"🍌 Image generated and cached: {public_url}")
        return public_url
        
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return None

def answer_question(question: str, history: str, subject: str, student_name: str, grade_level: str, country: str = "Unknown", board: str = "Unknown", mode: str = "tutoring", current_topic: str = "", allowed_subtopics: list = []) -> str:
    
    # 🚨 -1. SAFETY FIRST: Check for Crisis
    if detect_crisis_intent(question):
        print("🚨 Handling Crisis Response...")
        try:
            crisis_formatted = crisis_handling_prompt.format(
                student_name=student_name,
                question=question
            )
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(crisis_formatted)
            return response.text
        except Exception as e:
            print(f"❌ Error generating crisis response: {e}")
            return "I'm really sorry you're feeling this way. Please reach out to a trusted adult or emergency services immediately. You matter."

    # 0. Check Relevance (Critical Step)
    if mode == "tutoring" and current_topic:
        is_relevant = check_relevance(question, subject, current_topic, allowed_subtopics)
        if not is_relevant:
            # ✨ GOLDEN REDIRECTION RESPONSE
            return (
                f"😊 That’s an interesting question, {student_name}! "
                f"But right now, let's stay focused on **{current_topic}** so you can master it fully. 🎯\n\n"
                f"👉 Would you like me to explain more about {current_topic} or try a practice question?"
            )

    # 0.2 Check for Image Generation Intent
    if mode == "tutoring" and any(kw in question.lower() for kw in ["show me", "draw", "visualize", "image of", "picture of"]):
        print(f"🎨 Image request detected: {question}")
        image_topic = current_topic if current_topic else subject
        image_url = generate_cached_image(image_topic, grade_level)
        if image_url:
            return f"Here is a visualization for **{image_topic}**:\n\n![{image_topic}]({image_url})\n\nIs this helpful?"

    # 0.5 Check Response Cache (if not quiz mode)
    if mode == "tutoring":
        cached_response = check_response_cache(question, country, board, grade_level, subject)
        if cached_response:
            return cached_response
    
    # 1. Setup Retrieval from Pinecone
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # 2. Retrieve Context (Direct Pinecone Query with Multi-Level Filtering)
    relevant_context = ""
    retrieved_sources = []  # Track sources for citation
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embedding for the question
        question_embedding = embedding.embed_query(question)
        
        # 🎯 Build multi-level filter for curriculum alignment
        search_filter = {
            "subject": subject.capitalize()  # Match 'Physics', 'Math', etc.
        }
        
        # Add grade-level filter if available
        if grade_level and grade_level.lower() not in ['unknown', 'none', '']:
            search_filter["grade"] = grade_level
            print(f"🎓 Filtering by grade: {grade_level}")
        
        # Add board filter if available
        if board and board.lower() not in ['unknown', 'none', '']:
            search_filter["board"] = board
            print(f"📚 Filtering by board: {board}")
        
        # Add country filter if available
        if country and country.lower() not in ['unknown', 'none', '']:
            search_filter["country"] = country
            print(f"🌍 Filtering by country: {country}")
        
        print(f"🔍 Search filter: {search_filter}")
        
        # Query Pinecone directly with enhanced filtering
        query_response = index.query(
            vector=question_embedding,
            top_k=MAX_RETRIEVAL_CANDIDATES,
            include_metadata=True,
            filter=search_filter
        )
        
        # Filter documents by similarity score threshold
        relevant_docs = []
        for match in query_response.matches:
            similarity_score = (1 + match.score) / 2  # Convert from [-1,1] to [0,1]
            if similarity_score >= SIMILARITY_THRESHOLD:
                text_content = match.metadata.get('text', '')
                if text_content:
                    # Extract source information for citation
                    source_info = {
                        'chapter': match.metadata.get('chapter', 'Unknown Chapter'),
                        'topic': match.metadata.get('topic', ''),
                        'page': match.metadata.get('page', ''),
                        'score': similarity_score
                    }
                    retrieved_sources.append(source_info)
                    
                    relevant_docs.append({
                        'content': text_content,
                        'score': similarity_score,
                        'source': source_info
                    })
        
        if len(relevant_docs) > 0:
            relevant_context = "\n".join([doc['content'] for doc in relevant_docs])
            print(f"📖 Using context from {len(relevant_docs)} high-quality documents (Grade: {grade_level}, Board: {board})")
        else:
            print(f"⚠️ No relevant documents found in knowledge base for {subject} (Grade: {grade_level}, Board: {board})")
            # Return early with "not in textbook" message
            return (
                f"I apologize, {student_name}, but I couldn't find information about this topic in your current {subject} textbook "
                f"for {grade_level} ({board} curriculum). 📚\n\n"
                f"Would you like me to:\n"
                f"1. Suggest related topics we can explore?\n"
                f"2. Help you with a different {subject} question?"
            )
            
    except Exception as e:
        print(f"Retrieval error: {e}")
        relevant_context = ""

    # 3. Construct Prompt
    if mode == "intro":
        base_prompt = intro_prompt
    elif mode == "quiz":
        base_prompt = quiz_prompt
    else:
        base_prompt = tutoring_prompt

    # Logic to fill prompt variables
    input_variables = {
        "student_name": student_name, 
        "grade_level": grade_level, 
        "subject": subject,
        "input": question,
        "history": history,
        "context": relevant_context
    }
    
    # Special handling for quiz prompt which needs 'topic'
    if mode == "quiz":
        input_variables["topic"] = current_topic if current_topic else "this topic"

    formatted_prompt = base_prompt.format(**input_variables)

    # 4. Generate Response (Try Gemini First)
    generated_text = ""
    try:
        print("🤖 Attempting generation with Gemini...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(formatted_prompt)
        text = response.text
        
        if text:
            print("✅ Gemini generation successful")
            generated_text = text
            
    except Exception as gemini_err:
        print(f"❌ Gemini generation failed: {gemini_err}")
        print("🔄 Falling back to OpenAI...")

    # 5. Fallback to OpenAI
    if not generated_text:
        try:
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
            # We can just pass the formatted prompt as a single user message to ChatOpenAI
            # OR re-use the chain logic if we prefer structurally, but passing prompt is simpler here
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=formatted_prompt)]
            result = llm.invoke(messages)
            
            generated_text = str(result.content)

        except Exception as openai_err:
            print(f"❌ OpenAI generation failed: {openai_err}")
            return "I apologize, but I am currently unable to process your request. Please try again later."

    # 6. Add citation if not already present
    if mode == "tutoring" and generated_text and retrieved_sources:
        # Check if response already has a citation
        if "📖 Source:" not in generated_text and "Source:" not in generated_text:
            # Add citation from the highest-scoring source
            best_source = retrieved_sources[0]
            citation = f"\n\n📖 Source: {best_source['chapter']}"
            if best_source['topic']:
                citation += f" - {best_source['topic']}"
            if best_source['page']:
                citation += f" (Page {best_source['page']})"
            
            generated_text += citation
            print(f"✅ Added citation: {citation.strip()}")

    # 7. Cache the Response (only in tutoring mode)
    if mode == "tutoring" and generated_text:
        cache_response(question, generated_text, country, board, grade_level, subject)

    return generated_text

def suggest_topics_with_ai(subject: str, grade: str, student_name: str = "", student_id: str = "") -> list:
    fallback_topics = {
        "physics": ["Motion", "Force", "Work and Energy"],
        "math": ["Algebra", "Geometry", "Trigonometry"],
        "biology": ["Cells", "Genetics", "Evolution"]
    }

    try:
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedding,
            pinecone_api_key=PINECONE_API_KEY,
            namespace="default"
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": {
                    "student": student_id,
                    "source": "chat-history",
                    "type": "interaction"
                }
            }
        )

        result_docs = retriever.invoke(f"topics discussed in {subject}")
        extracted_topics = []

        for doc in result_docs:
            for line in doc.page_content.split("\n"):
                if subject.lower() in line.lower() or any(t in line.lower() for t in ["learn", "study", "explain"]):
                    words = line.split()
                    extracted_topics.extend([w.strip(".,") for w in words if len(w) > 3 and w[0].isupper()])

        cleaned = list(set([t.capitalize() for t in extracted_topics if t.isalpha()]))
        if len(cleaned) >= 3:
            print(f"📦 Suggested from Pinecone: {cleaned[:5]}")
            return cleaned[:5]

    except Exception as e:
        print(f"⚠️ Pinecone topic fetch failed: {e}")

    try:
        topic_prompt = PromptTemplate.from_template(
            "List 5 curriculum-based topics typically taught in {subject} for a student at the {grade} level. "
            "Return only the topic names, comma-separated, no extra text."
        )
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.4)
        chain = topic_prompt.partial(subject=subject, grade=grade) | llm
        result = chain.invoke({})
        raw_output = str(result.content if hasattr(result, "content") else result)
        print(f"🧠 Suggested from OpenAI: {raw_output}")

        topics = [topic.strip() for topic in raw_output.split(",") if topic.strip()]
        if len(topics) >= 2:
            return topics[:5]

    except Exception as e:
        print(f"⚠️ OpenAI topic suggestion error: {e}")

    return fallback_topics.get(subject.lower(), ["General Topic 1", "Topic 2", "Topic 3"])

def get_user_chat_history(student_id: str, subject: str):
    """Retrieve user's chat history from Pinecone"""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Query for chat history with this student and subject
        query_response = index.query(
            vector=[0.1] * 1536,  # Dummy vector for metadata search
            top_k=20,  # Get more results
            include_metadata=True,
            filter={
                "student": student_id,
                "source": "chat-history",
                "type": "interaction"
            }
        )
        
        # Extract chat history from metadata
        chat_history = []
        for match in query_response.matches:
            if 'text' in match.metadata:
                # Check if this is for the right subject
                entry_subject = match.metadata.get('subject', '').lower()
                if entry_subject == subject.lower():
                    chat_history.append({
                        'text': match.metadata['text'],
                        'timestamp': match.metadata.get('timestamp', '')
                    })
        
        # Sort by timestamp if available
        chat_history.sort(key=lambda x: x.get('timestamp', ''))
        
        # Extract just the text for return
        chat_history = [entry['text'] for entry in chat_history]
        
        print(f"📚 Retrieved {len(chat_history)} chat history entries for {student_id} in {subject}")
        return chat_history
        
    except Exception as e:
        print(f"⚠️ Error retrieving chat history: {e}")
        return []

def generate_prompt_suggestions(question: str, response: str, subject: str, student_name: str, grade_level: str, history: str = "") -> list:
    """
    Generate 3 contextual prompt suggestions based on the current conversation
    """
    try:
        # Create a prompt for generating follow-up questions
        suggestion_prompt = PromptTemplate.from_template("""
You are an AI tutor helping {student_name}, a {grade_level} student learning {subject}.

Based on the student's question: "{question}"
And your response: "{response}"

Generate exactly 3 follow-up questions or prompts that would help the student:
1. Deepen their understanding of the current topic
2. Explore related concepts
3. Apply what they've learned

Make the suggestions:
- Specific and actionable
- Appropriate for {grade_level} level
- Related to {subject}
- Encouraging and engaging
- 10-15 words each maximum

Format as a simple list, one per line, no numbering or bullets.

Examples of good suggestions:
- "Can you explain this with a real-world example?"
- "What happens if we change this variable?"
- "How does this relate to what we learned earlier?"
- "Can you give me a practice problem?"
- "What are the common mistakes students make here?"

Suggestions:
""")

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
        chain = suggestion_prompt.partial(
            student_name=student_name,
            grade_level=grade_level,
            subject=subject
        ) | llm

        result = chain.invoke({
            "question": question,
            "response": response
        })

        raw_suggestions = str(result.content if hasattr(result, "content") else result)
        
        # Parse the suggestions
        suggestions = []
        for line in raw_suggestions.split('\\n'):
            line = line.strip()
            if line and not line.startswith(('Suggestions:', 'Examples:', 'Format')):
                # Remove any numbering or bullets
                line = line.lstrip('123456789.-•* ')
                if len(line) > 5 and len(line) < 100:  # Reasonable length
                    suggestions.append(line)
        
        # Ensure we have exactly 3 suggestions
        if len(suggestions) >= 3:
            return suggestions[:3]
        elif len(suggestions) > 0:
            # Fill with fallback suggestions if needed
            fallbacks = [
                "Can you explain this concept with an example?",
                "How does this apply to real life?",
                f"What should I study next in {subject}?"
            ]
            while len(suggestions) < 3:
                suggestions.append(fallbacks[len(suggestions) % len(fallbacks)])
            return suggestions[:3]
        else:
            # Fallback suggestions if AI generation fails
            return [
                "Can you explain this with a real-world example?",
                f"How does this relate to other {subject} concepts?",
                "Can you give me a practice problem on this topic?"
            ]

    except Exception as e:
        print(f"⚠️ Error generating prompt suggestions: {e}")
        # Return fallback suggestions
        return [
            "Can you explain this with an example?",
            "How does this work in practice?",
            f"What should I study next in {subject}?"
        ]

def check_question_similarity(new_question: str, previous_questions: list, similarity_threshold: float = 0.85) -> tuple:
    """
    Check if a new question is similar to previous questions using embeddings.
    
    Returns:
        tuple: (is_similar: bool, similar_count: int, most_similar_question: str)
    """
    if not previous_questions:
        return False, 0, None
    
    try:
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        new_question_embedding = embedding.embed_query(new_question.lower())
        
        similar_count = 0
        most_similar_question = None
        highest_similarity = 0
        

        # Embed previous questions and compare
        for prev_q in previous_questions:
            prev_q_embedding = embedding.embed_query(prev_q.lower())
            
            # Calculate cosine similarity
            similarity = np.dot(new_question_embedding, prev_q_embedding) / (
                np.linalg.norm(new_question_embedding) * np.linalg.norm(prev_q_embedding)
            )
            
            if similarity >= similarity_threshold:
                similar_count += 1
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_question = prev_q
        
        is_similar = similar_count > 0
        return is_similar, similar_count, most_similar_question
        
    except Exception as e:
        print(f"⚠️ Error checking question similarity: {e}")
        # Fallback: simple text similarity check
        new_question_lower = new_question.lower().strip()
        similar_count = 0
        most_similar_question = None
        
        for prev_q in previous_questions:
            prev_q_lower = prev_q.lower().strip()
            # Check exact match or very high word overlap
            if prev_q_lower == new_question_lower:
                similar_count += 1
                most_similar_question = prev_q
            else:
                # Simple word overlap check
                new_words = set(new_question_lower.split())
                prev_words = set(prev_q_lower.split())
                if len(new_words) > 0 and len(prev_words) > 0:
                    overlap = len(new_words & prev_words) / max(len(new_words), len(prev_words))
                    if overlap >= 0.7:  # 70% word overlap
                        similar_count += 1
                        if most_similar_question is None:
                            most_similar_question = prev_q
        
        return similar_count > 0, similar_count, most_similar_question

def check_response_cache(question: str, country: str, board: str, grade: str, subject: str) -> str:
    """
    Check if a similar question has been answered before for same country/board/grade.
    Returns cached response if similarity >= 0.92, else None.
    """
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        question_embedding = embedding.embed_query(question)

        # Query Pinecone for cached responses
        query_response = index.query(
            vector=question_embedding,
            top_k=1,
            include_metadata=True,
            filter={
                "type": "response_cache",
                "country": country,
                "board": board,
                "grade": grade,
                "subject": subject.lower()
            }
        )

        if query_response.matches:
            match = query_response.matches[0]
            if match.score >= 0.92:
                print(f"✅ Found cached response (Score: {match.score:.4f})")
                return match.metadata.get('response_text')
            
    except Exception as e:
        print(f"⚠️ Cache check failed: {e}")
    
    return None

def cache_response(question: str, response: str, country: str, board: str, grade: str, subject: str):
    """
    Store question-response pair in Pinecone with metadata for future reuse.
    """
    try:
        if not response or len(response) < 10:
            return  # Don't cache empty or too short responses

        import uuid
        from datetime import datetime
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        question_embedding = embedding.embed_query(question)

        vector = {
            "id": f"cache_{str(uuid.uuid4())}",
            "values": question_embedding,
            "metadata": {
                "type": "response_cache",
                "country": country,
                "board": board,
                "grade": grade,
                "subject": subject.lower(),
                "question_text": question,
                "response_text": response,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        index.upsert(vectors=[vector])
        print("💾 Response cached for future use")

    except Exception as e:
        print(f"⚠️ Failed to cache response: {e}")


def retrieve_quiz_context(chapter_title: str, topics: list, subject: str) -> str:
    """
    Retrieve relevant context from the vector DB for quiz generation.
    Specifically looks for:
    1. Chapter summary/content (using chapter title and topics).
    2. Past papers or exam questions related to the chapter.
    """
    context_parts = []
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # 1. Search for General Chapter Content
        # Combine title and top 3 topics for a good search query
        query_text = f"{chapter_title} {' '.join(topics[:3])} summary concepts"
        query_vec = embedding.embed_query(query_text)
        
        content_results = index.query(
            vector=query_vec,
            top_k=5,
            include_metadata=True,
            filter={"subject": subject.capitalize()}
        )
        
        chapter_content = []
        for match in content_results.matches:
            if match.score > 0.75: # Only high relevance
                 chapter_content.append(match.metadata.get('text', ''))
        
        if chapter_content:
            context_parts.append("--- TEXTBOOK CONTENT ---")
            context_parts.extend(chapter_content)

        # 2. Search SPECIFICALLY for Past Papers / Exams
        # We assume past papers might be tagged or contain keywords like "Past Paper", "Exam", "Questions"
        past_paper_query = f"{chapter_title} past paper exam questions"
        pp_vec = embedding.embed_query(past_paper_query)
        
        pp_results = index.query(
            vector=pp_vec,
            top_k=5,
            include_metadata=True,
            filter={
                "subject": subject.capitalize(),
                # We optionally filter if we have a type field, but text search is broad
            }
        )
        
        past_papers = []
        for match in pp_results.matches:
            text = match.metadata.get('text', '')
            # Simple keyword check to boost confidence it's a question bank
            if any(k in text.lower() for k in ["question", "mark", "exam", "paper", "(a)", "(b)"]):
                past_papers.append(text)
        
        if past_papers:
            print(f"📄 Found {len(past_papers)} potential past paper fragments for {chapter_title}")
            context_parts.append("\\n--- PAST PAPER QUESTIONS (PRIORITIZE THESE) ---")
            context_parts.extend(past_papers)
            
    except Exception as e:
        print(f"❌ Error retrieving quiz context: {e}")
        
    return "\\n\\n".join(context_parts)
