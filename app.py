from fastapi import FastAPI, UploadFile, Form
from embed import embed_pdf
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.post("/embed")
async def upload_pdf(
    file: UploadFile,
    grade: str = Form(...),
    subject: str = Form(...),
    curriculum: str = Form(...)
):
    contents = await file.read()
    filename = file.filename
    save_path = f"uploads/{filename}"

    with open(save_path, "wb") as f:
        f.write(contents)

    result = embed_pdf(save_path, {
        "grade": grade,
        "subject": subject,
        "curriculum": curriculum
    })

    return result
