import os
import logging
from set_env import set_huggingface_token
from utils.audio_processing import extract_audio_from_media, save_target_speaker_audio
from utils.speaker_identification import get_reference_embedding, diarize_and_identify
from utils.transcription import transcribe_audio
from utils.analysis import analyze_audio_features, analyze_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_media_with_reference(media_path, reference_audio_path):
    temp_audio_path = "temp_audio.wav"
    target_speaker_audio_path = "target_speaker.wav"

    # Step 1: Extract Audio from Media
    is_audio = media_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'))
    if is_audio:
        logger.info(f"Copying audio file {media_path} to {temp_audio_path}...")
        os.system(f'cp "{media_path}" "{temp_audio_path}"')
    else:
        logger.info(f"Extracting audio from {media_path} to {temp_audio_path}...")
        extract_audio_from_media(media_path, temp_audio_path)

    # Step 2: Get Reference Embedding
    logger.info("Computing reference embedding...")
    reference_embedding = get_reference_embedding(reference_audio_path)

    # Step 3: Diarize and Identify Target Speaker
    logger.info("Diarizing and identifying target speaker...")
    target_speaker_segments = diarize_and_identify(temp_audio_path, reference_embedding)

    # Step 4: Save Target Speaker Audio
    if target_speaker_segments:
        logger.info("Saving target speaker audio segments...")
        save_target_speaker_audio(temp_audio_path, target_speaker_segments, target_speaker_audio_path)
    else:
        logger.warning(f"No segments found for the target speaker in {media_path}.")
        os.remove(temp_audio_path)
        return None

    # Step 5: Transcribe Audio
    logger.info("Transcribing target speaker audio...")
    transcription = transcribe_audio(target_speaker_audio_path)

    # Step 6: Analyze Audio Features
    logger.info("Analyzing audio features...")
    audio_features = analyze_audio_features(target_speaker_audio_path, transcription)

    # Step 7: Advanced Text Analysis
    logger.info("Performing advanced text analysis...")
    text_analysis = analyze_text(transcription)

    # Combine Results
    analysis_document = {
        'transcription': transcription,
        'audio_features': audio_features,
        'text_analysis': text_analysis
    }

    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(target_speaker_audio_path)

    return analysis_document

if __name__ == "__main__":
    # Step 0: Set Hugging Face token from file using set_env
    set_huggingface_token()

    # Use shorter audio files for testing to reduce processing time
    media_files = ["temp_audio2.mp3"]  # Replace with your media file
    reference_audio = "reference_audio.mp3"  # Path to your reference audio file

    for media in media_files:
        logger.info(f"Processing {media}...")
        analysis = process_media_with_reference(media, reference_audio)
        if analysis:
            logger.info(f"Analysis for {media}:")
            logger.info(analysis)
        else:
            logger.warning(f"Could not find target speaker in {media}.")
