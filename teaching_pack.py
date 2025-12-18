import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from schemas import TeachingPack
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    # Prefer OpenAI for structured JSON, fallback to Gemini if configured
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o", temperature=0.1)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

def generate_teaching_pack(file_path: str) -> dict:
    # 1. Extract Text
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])
    else:
        # Fallback for text files (like our test case)
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    
    # 2. Setup Parser
    parser = PydanticOutputParser(pydantic_object=TeachingPack)

    # 3. Setup Prompt
    prompt = PromptTemplate(
        template="""You are an expert AI Curriculum Developer. 
        Your task is to analyze the following textbook content and generate a comprehensive, exam-oriented "Teaching Pack".
        
        The output must strictly follow the JSON schema provided.
        
        Key Requirements:
        1. **Chapter Profile**: Extract learning objectives and break down content types (theory vs numericals).
        2. **Sessions**: Create a logical sequence of teaching sessions. Each session must have teach points, checks, and practice items.
        3. **Formulas & Numericals**: If the text contains formulas, explicitly populate the 'formulas' and 'worked_examples' sections.
        4. **Question Bank**: Generate a diverse set of questions (MCQ, Short, Numeric) mapped to topics.
        5. **Exam Focus**: If you detect emphasis on certain topics, reflect that in 'topic_exam_weighting' (simulate if no actual past paper data is available in the provided text).
        
        {format_instructions}

        TEXTBOOK CONTENT:
        {text}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 4. Run Chain
    llm = get_llm()
    chain = prompt | llm | parser

    try:
        # Depending on context limit, we might need to truncate text
        # GPT-4o has 128k context, Gemini 1.5 has 1M+. We should be safe with most chapters.
        result = chain.invoke({"text": full_text})
        return result.dict()
    except Exception as e:
        print(f"Error generating teaching pack: {e}")
        raise e

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        pack = generate_teaching_pack(sys.argv[1])
        print(pack)
