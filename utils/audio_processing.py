import subprocess
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def extract_audio_from_media(media_path, audio_path):
    # Use FFmpeg to extract or convert audio
    logger.info(f"Extracting audio from {media_path} to {audio_path}...")
    command = [
        'ffmpeg',
        '-i', media_path,
        '-q:a', '0',
        '-map', 'a',
        '-y',  # Overwrite output files
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def save_target_speaker_audio(audio_path, segments, output_path):
    audio_format = audio_path.split('.')[-1].lower()
    audio = AudioSegment.from_file(audio_path, format=audio_format)
    target_speaker_audio = AudioSegment.empty()
    for segment in segments:
        start_ms = segment.start * 1000
        end_ms = segment.end * 1000
        target_speaker_audio += audio[start_ms:end_ms]
    target_speaker_audio.export(output_path, format="wav")
    logger.info(f"Saved target speaker audio to {output_path}.")
