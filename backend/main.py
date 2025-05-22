import subprocess
import sys

# Ensure all dependencies are installed
required_packages = [
    "fastapi",
    "uvicorn",
    "openai",
    "textract",
    "langdetect",
    "googletrans==4.0.0-rc1",
    "fpdf",
    "python-multipart"
]

for package in required_packages:
    try:
        __import__(package.split('==')[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from fastapi import FastAPI, UploadFile, Form, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import openai
import os
import textract
import mimetypes
from langdetect import detect
from googletrans import Translator
from fpdf import FPDF
import uuid

# Load your API key from environment variables or config
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
translator = Translator()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for job description input
class TailorRequest(BaseModel):
    resume_text: str
    job_description: str
    force_language: Optional[str] = None  # Language override option
    export_pdf: Optional[bool] = False    # Option to export PDF

# Health check endpoint for frontend readiness
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Endpoint for uploading resume file and extracting text
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()
        supported_exts = ['.pdf', '.docx', '.doc', '.txt', '.rtf']

        if ext not in supported_exts:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file format: {ext}. Supported formats: {supported_exts}"})

        temp_filename = f"temp_{uuid.uuid4()}{ext}"
        with open(temp_filename, "wb") as f:
            f.write(contents)
        text = textract.process(temp_filename).decode("utf-8")
        os.remove(temp_filename)

        language = detect(text)
        return {"resume_text": text, "language": language}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Tailor resume using GPT-4
@app.post("/tailor-resume")
async def tailor_resume(request: TailorRequest):
    try:
        resume_language = request.force_language or detect(request.resume_text)
        job_language = detect(request.job_description)

        resume_text_en = translator.translate(request.resume_text, src=resume_language, dest='en').text if resume_language != 'en' else request.resume_text
        job_description_en = translator.translate(request.job_description, src=job_language, dest='en').text if job_language != 'en' else request.job_description

        prompt = f"""
        You are a professional career assistant. Given the resume and the job description below,
        tailor the resume to better match the job requirements. Focus on the summary, experience,
        and skills sections.

        Resume:
        {resume_text_en}

        Job Description:
        {job_description_en}

        Provide the updated resume in English:
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful resume tailoring assistant. Please respond in English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        tailored_resume_en = response['choices'][0]['message']['content']

        # Translate back to original resume language if needed
        tailored_resume_final = translator.translate(tailored_resume_en, src='en', dest=resume_language).text if resume_language != 'en' else tailored_resume_en

        # If export to PDF is requested
        if request.export_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)

            for line in tailored_resume_final.split('\n'):
                pdf.multi_cell(0, 10, line)

            pdf_output_path = f"tailored_resume_{uuid.uuid4()}.pdf"
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as f:
                pdf_data = f.read()
            os.remove(pdf_output_path)

            return Response(content=pdf_data, media_type="application/pdf", headers={
                "Content-Disposition": "attachment; filename=tailored_resume.pdf"
            })

        return {"tailored_resume": tailored_resume_final, "language": resume_language}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
