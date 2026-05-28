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

# ---------------- ENV ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You are an expert HR recruiter and senior software engineer.

Analyze and rank a developer resume.

Return ONLY valid JSON:

{
  "candidate_level": "Beginner/Intermediate/Advanced",
  "score": 0,
  "strengths": [],
  "weaknesses": [],
  "recommended_roles": [],
  "final_verdict": ""
}

Rules:
- Score must be 0–100
- Return ONLY JSON
"""

# ---------------- HELPERS ----------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,:/@+\-#()]", "", text)
    return text.strip()


def extract_pdf_text(file_path: str) -> str:
    return extract_text(file_path)


# ---------------- ROUTES ----------------
@app.get("/")
async def root():
    return {"message": "AI Resume Ranker API Running"}


@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None

    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text
        resume_text = extract_pdf_text(tmp_path)
        cleaned = clean_text(resume_text)

        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Groq request
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this resume:\n\n{cleaned}"}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        result = response.choices[0].message.content.strip()

        # Parse JSON safely
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
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)