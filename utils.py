import torch
import librosa
import numpy as np
import json
import openai
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

import librosa.util.deprecation
setattr(librosa.util.deprecation, 'AUDIOREAD_PARAMS', {'backend': 'audioread'})

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.audio")

SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE

# Ensure OpenAI API key is set
# openai.api_key = os.getenv("OPENAI_API_KEY")

def load_audio(audio_path, sr=None):
    try:
        y, sr = librosa.load(audio_path, sr=sr, res_type='kaiser_fast')
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.zeros(SAMPLE_RATE), SAMPLE_RATE

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        print(f"Error transcribing audio with OpenAI: {e}")
        return ""

def analyze_accent_with_openai(audio_path, transcription):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional speech analyst. Analyze the accent in the transcription and provide a score from 0-10, where 10 is the most native-sounding accent."},
                {"role": "user", "content": f"Analyze the accent in this transcription: '{transcription}'. Provide only a numerical score between 0 and 10."}
            ],
            temperature=0.3,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            return round(min(10, max(0, score)), 2)
        except ValueError:
            return 5.0
    except Exception as e:
        print(f"Error analyzing accent with OpenAI: {e}")
        return 5.0

def analyze_clarity_with_openai(transcription):
    if not transcription:
        return 5.0
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional speech analyst. Analyze the clarity and articulation in the transcription and provide a score from 0-10, where 10 is perfectly clear and articulated."},
                {"role": "user", "content": f"Analyze the clarity and articulation in this transcription: '{transcription}'. Provide only a numerical score between 0 and 10."}
            ],
            temperature=0.3,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            return round(min(10, max(0, score)), 2)
        except ValueError:
            return 5.0
    except Exception as e:
        print(f"Error analyzing clarity with OpenAI: {e}")
        return 5.0

def analyze_confidence_with_openai(transcription):
    if not transcription:
        return 5.0
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional speech analyst. Analyze the confidence and tone in the transcription and provide a score from 0-10, where 10 is extremely confident."},
                {"role": "user", "content": f"Analyze the confidence and tone in this transcription: '{transcription}'. Provide only a numerical score between 0 and 10."}
            ],
            temperature=0.3,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            return round(min(10, max(0, score)), 2)
        except ValueError:
            return 5.0
    except Exception as e:
        print(f"Error analyzing confidence with OpenAI: {e}")
        return 5.0

def analyze_vocabulary_with_openai(transcription):
    if not transcription:
        return 5.0
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional speech analyst. Analyze the vocabulary and language use in the transcription and provide a score from 0-10, where 10 represents excellent vocabulary."},
                {"role": "user", "content": f"Analyze the vocabulary and language use in this transcription: '{transcription}'. Provide only a numerical score between 0 and 10."}
            ],
            temperature=0.3,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            return round(min(10, max(0, score)), 2)
        except ValueError:
            return 5.0
    except Exception as e:
        print(f"Error analyzing vocabulary with OpenAI: {e}")
        return 5.0

def match_accent_with_openai(transcription):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional linguistic analyst. Identify the accent in the transcription and provide a similarity score from 0-10, where 10 is the highest similarity."},
                {"role": "user", "content": f"Identify the accent in this transcription: '{transcription}'. Respond with a JSON object containing 'accent' and 'similarity_score' (0-10)."}
            ],
            temperature=0.3,
            max_tokens=100
        )
        result_text = response.choices[0].message.content.strip()
        try:
            result = json.loads(result_text)
            similarity_score = float(result.get("similarity_score", 0))
            return result.get("accent", "unknown"), round(min(10, max(0, similarity_score)) / 10, 2)
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing accent match result: {e}")
            return "unknown", 0.0
    except Exception as e:
        print(f"Error matching accent with OpenAI: {e}")
        return "unknown", 0.0

def analyze_speech(audio_path):
    transcription = transcribe_audio(audio_path)
    
    with ThreadPoolExecutor() as executor:
        future_accent = executor.submit(analyze_accent_with_openai, audio_path, transcription)
        future_clarity = executor.submit(analyze_clarity_with_openai, transcription)
        future_confidence = executor.submit(analyze_confidence_with_openai, transcription)
        future_vocabulary = executor.submit(analyze_vocabulary_with_openai, transcription)
        future_accent_match = executor.submit(match_accent_with_openai, transcription)
        
        accent_score = future_accent.result()
        clarity_score = future_clarity.result()
        confidence_score = future_confidence.result()
        vocabulary_score = future_vocabulary.result()
        matched_accent, similarity_score = future_accent_match.result()
    
    overall_score = round((accent_score + clarity_score + confidence_score + vocabulary_score) / 4, 2)
    overall_summary = get_overall_summary_with_openai(transcription, overall_score, 
                                                     accent_score, clarity_score, 
                                                     confidence_score, vocabulary_score)
    
    data = {
        "transcription": transcription,
        "scores": {
            "overall_score": overall_score,
            "overall_summary": overall_summary,
            "accent": {
                "score": accent_score,
                "matched_accent": matched_accent,
                "similarity_score": round(similarity_score*10, 2)
            },
            "clarity_and_articulation": {
                "score": clarity_score
            },
            "confidence_and_tone": {
                "score": confidence_score
            },
            "vocabulary_and_language_use": {
                "score": vocabulary_score
            },
        }
    }
    
    # Save results to file
    with open("results.json", "w") as f:
        json.dump(data, f, indent=4)
    
    return data

def get_overall_summary_with_openai(transcription, overall_score, accent_score, clarity_score, confidence_score, vocabulary_score):
    try:
        prompt = f"""
        Generate a concise summary of speech analysis with these scores:
        - Overall: {overall_score}/10
        - Accent: {accent_score}/10
        - Clarity: {clarity_score}/10
        - Confidence: {confidence_score}/10
        - Vocabulary: {vocabulary_score}/10
        
        Transcription: '{transcription}'
        
        Provide a constructive 2-3 sentence summary with strengths and areas for improvement.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional speech analyst providing constructive feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating overall summary with OpenAI: {e}")
        return "Unable to generate summary."