import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import List
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pydantic import BaseModel
from utils import analyze_speech
from functions import get_aws_questions
import asyncio
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GetAwsQns(BaseModel):
    company: str
    role: str

class QuestionsResponse(BaseModel):
    questions: List[str]

@app.post("/score")
async def score_language(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_speech, temp_audio_path)
        os.unlink(temp_audio_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/get_questions", response_model=QuestionsResponse)
async def generate_questions(data: GetAwsQns):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_aws_questions, data.company, data.role)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app:app", port=8080, reload=True)