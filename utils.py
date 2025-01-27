import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import whisper
import warnings
from functions import get_summary, get_overall_summary
import json

import librosa.util.deprecation
setattr(librosa.util.deprecation, 'AUDIOREAD_PARAMS', {'backend': 'audioread'})

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.audio")

SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
NUM_MFCC = 13
MAX_MFCC_LENGTH = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base").to(device)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

class AccentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AccentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (NUM_MFCC // 8) * (MAX_MFCC_LENGTH // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@torch.no_grad()
def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def load_audio(audio_path, sr=None):
    """Load audio file using audioread directly"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, res_type='kaiser_fast')
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.zeros(SAMPLE_RATE), SAMPLE_RATE

def analyze_accent(audio_path):
    y, sr = load_audio(audio_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_data = pitches[pitches > 0]
    if len(pitch_data) == 0:
        return 5.0
    pitch_mean = np.mean(pitch_data)
    pitch_std = np.std(pitch_data)
    accent_score = min(10, max(0, (pitch_std / pitch_mean) * 10))
    return round(accent_score, 2)

def analyze_clarity(text):
    if not text:
        return 5.0
    words = text.split()
    if not words:
        return 5.0
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_count = max(1, sum(1 for char in text if char in '.!?'))
    avg_sentence_length = len(words) / sentence_count
    clarity_score = 10 - min(10, max(0, (avg_word_length - 4) + (avg_sentence_length - 10)))
    return round(clarity_score, 2)

@torch.no_grad()
def analyze_confidence(text):
    if not text:
        return 5.0
    sentiment = sentiment_pipeline(text)[0]
    confidence_score = sentiment['score'] * 10 if sentiment['label'] == 'POSITIVE' else (1 - sentiment['score']) * 10
    return round(confidence_score, 2)

@torch.no_grad()
def analyze_vocabulary(text):
    if not text:
        return 5.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1)
    vocab_score = probabilities.max().item() * 10
    return round(vocab_score, 2)

def extract_features_mfcc(audio_path, n_mfcc=13):
    y, sr = load_audio(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def match_accent(input_audio, vector_json):
    try:
        with open(vector_json, 'r') as f:
            accent_vectors = json.load(f)
    except FileNotFoundError:
        return "unknown", 0.0

    input_vector = extract_features_mfcc(input_audio)
    input_vector = normalize([input_vector])[0]

    highest_similarity = -1
    matched_accent = "unknown"

    for accent, vectors in accent_vectors.items():
        for vector in vectors:
            vector = np.array(vector)
            vector = normalize([vector])[0]
            similarity = 1 - cosine(input_vector, vector)
            if similarity > highest_similarity:
                highest_similarity = similarity
                matched_accent = accent

    return matched_accent, highest_similarity

def analyze_speech(audio_path):
    from concurrent.futures import ThreadPoolExecutor
    
    transcription = transcribe_audio(audio_path)
    
    with ThreadPoolExecutor() as executor:
        future_accent = executor.submit(analyze_accent, audio_path)
        future_clarity = executor.submit(analyze_clarity, transcription)
        future_confidence = executor.submit(analyze_confidence, transcription)
        future_vocabulary = executor.submit(analyze_vocabulary, transcription)
        future_accent_match = executor.submit(match_accent, audio_path, 'accent_vectors.json')
        
        accent_score = future_accent.result()
        clarity_score = future_clarity.result()
        confidence_score = future_confidence.result()
        vocabulary_score = future_vocabulary.result()
        matched_accent, similarity_score = future_accent_match.result()
    
    overall_score = round((accent_score + clarity_score + confidence_score + vocabulary_score) / 4, 2)
    
    data = {
        "transcription": transcription,
        "scores": {
            "overall_score": overall_score,
            "overall_summary": "",
            "accent": {
                "score": accent_score,
                "matched_accent": matched_accent,
                "similarity_score": round(similarity_score*10,2)
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
    
    data["scores"]["overall_summary"] = get_overall_summary(data)
    
    with open("results.json", "w") as f:
        json.dump(data, f, indent=4)
    
    return data