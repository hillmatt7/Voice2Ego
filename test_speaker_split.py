import os
import torch
import librosa
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition
import logging
import torch.nn.functional as F

# Setup logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
reference_audio_path = "Friedland.mp3"  # Your reference audio file
test_audio_path = "Audio for Testing.mp3"  # Your test audio file

# Load the speaker verification model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/embedding_model"
)

# Function to get reference embedding
def get_reference_embedding(reference_audio_path):
    if not os.path.isfile(reference_audio_path):
        logger.error(f"Reference audio file not found: {reference_audio_path}")
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")
    
    logger.info("Creating reference embedding...")
    signal, fs = librosa.load(reference_audio_path, sr=16000)
    signal = torch.from_numpy(signal).unsqueeze(0)
    with torch.no_grad():
        embedding = verification.encode_batch(signal)
    return embedding.squeeze(0).detach()

# Function to perform diarization and identify speakers
def diarize_and_identify(audio_path, reference_embedding, threshold=0.65):
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token is None:
        logger.error("HUGGINGFACE_TOKEN environment variable not set.")
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
    
    # Load Pyannote pipeline
    logger.info("Loading pyannote.audio pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )

    # Perform diarization
    logger.info("Performing diarization...")
    diarization = pipeline(audio_path)
    logger.info("Diarization completed.")

    target_speaker_segments = []

    # Adjusted loop to match the new API
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        logger.debug(f"Processing segment: {segment}, speaker: {speaker}")

        # Load segment audio
        signal, fs = librosa.load(
            audio_path, sr=16000, offset=segment.start, duration=segment.duration
        )
        if len(signal) == 0:
            continue  # Skip empty segments

        signal = torch.from_numpy(signal).unsqueeze(0)
        with torch.no_grad():
            # Compute embedding for segment
            embedding = verification.encode_batch(signal).squeeze(0).detach()
        # Compute similarity score using cosine similarity
        score = F.cosine_similarity(reference_embedding, embedding, dim=0)
        # Determine if it's the target speaker
        if score.item() > threshold:
            logger.debug(f"Segment {segment} matches target speaker with score {score.item()}.")
            target_speaker_segments.append(segment)
        else:
            logger.debug(f"Segment {segment} does not match target speaker (score: {score.item()}).")

    logger.info(f"Identified {len(target_speaker_segments)} segments matching the reference speaker.")
    return target_speaker_segments

# Main test function
if __name__ == "__main__":
    try:
        # Step 1: Create reference embedding
        reference_embedding = get_reference_embedding(reference_audio_path)

        # Step 2: Test diarization and speaker identification
        result_segments = diarize_and_identify(test_audio_path, reference_embedding)

        # Log results
        logger.info(f"Result Segments: {result_segments}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
