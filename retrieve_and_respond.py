import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

intro_prompt = PromptTemplate.from_template("""
You are a cheerful, encouraging, and patient AI tutor named XRTutor. You help {student_name}, a {grade_level} learner, understand the topic of **{subject}**.

Start by introducing yourself and asking:
1. What would you like to learn about today?
2. Are you a high school student, college student, or a professional?
3. What do you already know about this topic?

After that, begin teaching.

Context:
{history}

Student said:
{input}

Answer:
""")

tutoring_prompt = PromptTemplate.from_template("""
You are a helpful AI tutor for {student_name}, a {grade_level} student learning about **{subject}**.

The student may be unsure about the topic. Begin by explaining the concept clearly with analogies or real-world examples.

If they seem confused or unsure:
- Encourage: "No problem! Let's explore this together."
- Ask: "If you'd like me to guide you step by step, just say yes."
- Use relatable examples.

End every few responses by asking if the student wants to try a short quiz.

Context:
{history}

Student said:
{input}

Answer:
""")

quiz_prompt = PromptTemplate.from_template("""
You are a quiz master AI helping {student_name}, a {grade_level} student in **{subject}**.

Create a 1-question multiple choice quiz about this topic. Only provide:
- The question
- 4 options labeled A, B, C, and D
- Ask student to pick one option

Context:
{history}

Student said:
{input}

Answer:
""")

def answer_question(question: str, history: str, subject: str, student_name: str, grade_level: str, mode: str = "tutoring") -> str:
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY,
        namespace="default"
    )

    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "filter": {
            "source": subject.lower(),
            "grade": grade_level
        }
    })

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

    if mode == "intro":
        chain = intro_prompt.partial(student_name=student_name, grade_level=grade_level, subject=subject) | llm
    elif mode == "quiz":
        chain = quiz_prompt.partial(student_name=student_name, grade_level=grade_level, subject=subject) | llm
    else:
        chain = tutoring_prompt.partial(student_name=student_name, grade_level=grade_level, subject=subject) | llm

    try:
        retrieved_docs = retriever.invoke(question)
        docs_found = bool(retrieved_docs)
    except Exception as e:
        print(f"Retrieval error: {e}")
        docs_found = False

    if docs_found:
        try:
            retrieval_chain = create_retrieval_chain(retriever, chain)
            result = retrieval_chain.invoke({
                "input": question,
                "history": history
            })
            answer = result["answer"]
        except Exception as e:
            print(f"Retrieval chain error: {e}")
            result = chain.invoke({
                "input": question,
                "history": history
            })
            answer = result
    else:
        result = chain.invoke({
            "input": question,
            "history": history
        })
        answer = result

    text = str(answer.content) if hasattr(answer, "content") else str(answer)

    if not text.strip() or "error" in text.lower():
        return "🛛 I'm unable to answer that right now. Try rephrasing or ask a different question."

    return text

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
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY,
        namespace="default"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "student": student_id,
                "source": "chat-history",
                "type": "interaction"
            }
        }
    )
    return retriever.invoke(f"past interactions in {subject}")
