from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text
from groq import Groq
from dotenv import load_dotenv
import os
import tempfile
import re
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are an expert HR recruiter and senior software engineer.

Your task is to analyze and rank a developer resume.

Evaluate the resume on:
1. Technical Skills
2. Experience
3. Projects
4. Communication
5. Problem Solving
6. Education
7. Modern Technologies
8. Overall Developer Quality

Return ONLY valid JSON in this format:

{
  "candidate_level": "Beginner/Intermediate/Advanced",
  "score": 0,
  "strengths": [
    "strength 1",
    "strength 2"
  ],
  "weaknesses": [
    "weakness 1",
    "weakness 2"
  ],
  "recommended_roles": [
    "role 1",
    "role 2"
  ],
  "final_verdict": "Short explanation"
}

Rules:
- Score must be between 0 and 100.
- Be strict but fair.
- Focus mainly on software engineering and developer quality.
- Return ONLY JSON.
"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,:/@+\-#()]", "", text)
    return text.strip()

def extract_pdf_text(file_path):
    return extract_text(file_path)

@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        resume_text = extract_pdf_text(tmp_path)
        cleaned = clean_text(resume_text)

        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Analyze this developer resume and rank it:\n\n{cleaned}"
                }
            ],
            temperature=0.3,
            max_tokens=1200
        )

        result = response.choices[0].message.content.strip()

        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Model did not return valid JSON"
            )

        return {
            "success": True,
            "filename": file.filename,
            "analysis": parsed_result
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
async def root():
    return {
        "message": "AI Resume Ranker API Running"
    }