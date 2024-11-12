import whisper
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

