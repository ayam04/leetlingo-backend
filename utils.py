import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import whisper
import os
import re
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from functions import get_summary, get_overall_summary


SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
NUM_MFCC = 13
MAX_MFCC_LENGTH = 1000


whisper_model = whisper.load_model("base")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")


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

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def analyze_accent(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])
    pitch_std = np.std(pitches[pitches > 0])
    
    accent_score = min(10, max(0, (pitch_std / pitch_mean) * 10))
    return round(accent_score, 2)

def analyze_clarity(text):
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = len(words) / max(1, sentence_count)
    
    clarity_score = 10 - min(10, max(0, (avg_word_length - 4) + (avg_sentence_length - 10)))
    return round(clarity_score, 2)

def analyze_confidence(text):
    sentiment = sentiment_pipeline(text)[0]
    confidence_score = sentiment['score'] * 10 if sentiment['label'] == 'POSITIVE' else (1 - sentiment['score']) * 10
    return round(confidence_score, 2)

def analyze_vocabulary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    vocab_score = probabilities.max().item() * 10
    return round(vocab_score, 2)

def analyze_pronunciation(transcript, accent_patterns):
    accent_scores = defaultdict(int)
    
    for accent, patterns in accent_patterns.items():
        for pattern, _ in patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            accent_scores[accent] += len(matches)
    
    return accent_scores

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_AUDIO_LENGTH/SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    
    if mfcc.shape[1] > MAX_MFCC_LENGTH:
        mfcc = mfcc[:, :MAX_MFCC_LENGTH]
    else:
        pad_width = ((0, 0), (0, MAX_MFCC_LENGTH - mfcc.shape[1]))
        mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
    
    return torch.FloatTensor(mfcc).unsqueeze(0)

def extract_features_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def process_audio_files(audio_folder, output_json):
    data = {}
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):
            label = file_name.split("_")[0]
            file_path = os.path.join(audio_folder, file_name)
            vector = extract_features_mfcc(file_path).tolist()

            if label in data:
                data[label].append(vector)
            else:
                data[label] = [vector]
    
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Vector saved: {output_json}")

def match_accent(input_audio, vector_json):
    with open(vector_json, 'r') as f:
        accent_vectors = json.load(f)

    input_vector = extract_features_mfcc(input_audio)
    input_vector = normalize([input_vector])[0]

    highest_similarity = -1
    matched_accent = None

    for accent, vectors in accent_vectors.items():
        for vector in vectors:
            vector = np.array(vector)
            vector = normalize([vector])[0]
            similarity = 1 - cosine(input_vector, vector)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                matched_accent = accent

    return matched_accent, highest_similarity

def load_label_encoder(meta_file):
    meta_data = pd.read_csv(meta_file)
    le = LabelEncoder()
    le.fit(meta_data['primary_language'])
    return le 

def load_model(model_path, num_classes):
    model = AccentClassifier(num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def analyze_speech(audio_path):
    transcription = transcribe_audio(audio_path)
    
    accent_score = analyze_accent(audio_path)
    clarity_score = analyze_clarity(transcription)
    confidence_score = analyze_confidence(transcription)
    vocabulary_score = analyze_vocabulary(transcription)
    matched_accent, similarity_score = match_accent(audio_path, 'accent_vectors.json')
    
    overall_score = round((accent_score + clarity_score + confidence_score + vocabulary_score) / 4, 2)
    
    data = {
        "transcription": transcription,
        "scores": {
            "overall_score": overall_score,
            "overall_summary": "",
            "accent": {
                "score": accent_score,
                "summary": get_summary("accent", accent_score, transcription),
                "matched_accent": matched_accent,
                "similarity_score": round(similarity_score*10,2)
            },
            "clarity_and_articulation": {
                "score": clarity_score,
                "summary": get_summary("clarity and articulation", clarity_score, transcription)
            },
            "confidence_and_tone": {
                "score": confidence_score,
                "summary": get_summary("confidence and tone", confidence_score, transcription)
            },
            "vocabulary_and_language_use": {
                "score": vocabulary_score,
                "summary": get_summary("vocabulary and language use", vocabulary_score, transcription)
            },
        }
    }

    data["scores"]["overall_summary"] = get_overall_summary(data)

    return data

# if __name__ == "__main__":
#     audio_file_path = "AudioSamples/accent_test.wav"
#     results = analyze_speech(audio_file_path)
#     print(json.dumps(results, indent=2))