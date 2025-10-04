from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from retrieve_and_respond import answer_question  # must exist

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    chat_history: Optional[List[str]] = []

@app.post("/ask")
def ask(request: AskRequest):
    response = answer_question(request.question, history="\n".join(request.chat_history))
    return {
        "student": "Saad",
        "answer": response
    }
