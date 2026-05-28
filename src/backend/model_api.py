from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from groq import Groq
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
Return ONLY valid JSON.

{
  "candidate_level": "Beginner/Intermediate/Advanced",
  "score": 0,
  "strengths": [],
  "weaknesses": [],
  "recommended_roles": [],
  "final_verdict": ""
}

Score must be 0 to 100.
No text. No markdown. Only JSON.
"""

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,:/@+\-#()]", "", text)
    return text.strip()

def extract_pdf_text(file_path: str) -> str:
    return extract_text(file_path)

def extract_json(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start:end+1]

@app.get("/")
async def root():
    return {"message": "AI Resume Ranker API Running"}

@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        resume_text = extract_pdf_text(tmp_path)
        cleaned = clean_text(resume_text)

        if not cleaned:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": cleaned}
            ],
            temperature=0
        )

        result = response.choices[0].message.content
        json_str = extract_json(result)

        if not json_str:
            raise HTTPException(status_code=500, detail="Invalid model response")

        parsed_result = json.loads(json_str)

        return {
            "success": True,
            "filename": file.filename,
            "analysis": parsed_result
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)