import librosa
import numpy as np
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

def analyze_audio_features(audio_path, transcription):
    y, sr = librosa.load(audio_path)

    # Speed (Speaking Rate)
    duration = librosa.get_duration(y=y, sr=sr)
    word_count = len(transcription.split())
    speaking_rate = word_count / duration  # words per second

    # Cadence (Rhythmic Flow)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Tone (Pitch Analysis)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    average_pitch = np.mean(pitches) if len(pitches) > 0 else 0

    return {
        'speaking_rate': speaking_rate,
        'tempo': tempo,
        'average_pitch': average_pitch
    }

def analyze_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(text)

    return {
        'summary': summary[0]['summary_text'],
        'sentiment': sentiment
    }
