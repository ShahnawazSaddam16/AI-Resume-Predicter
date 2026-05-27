from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pdfminer.high_level import extract_text
import os
import tempfile
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(r"C:\\ML Projects\\AI Resume Predicter\\resume_classifier.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

def extract_pdf_text(file_path):
    text = extract_text(file_path)
    return text

@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    resume_text = extract_pdf_text(tmp_path)

    os.remove(tmp_path)

    cleaned = clean_text(resume_text)

    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities)) * 100

    return {
        "category": prediction,
        "confidence": round(confidence, 2)
    }