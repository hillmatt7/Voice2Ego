import os
import torch
import logging
import librosa
import torch.nn.functional as F
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition

logger = logging.getLogger(__name__)

def get_reference_embedding(reference_audio_path):
    if not os.path.isfile(reference_audio_path):
        logger.error(f"Reference audio file not found: {reference_audio_path}")
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

    logger.info("Creating reference embedding...")

    # Load audio using librosa
    signal, fs = librosa.load(reference_audio_path, sr=16000)

    # Convert to PyTorch tensor
    signal = torch.from_numpy(signal).unsqueeze(0)

    # Load verification model
    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/speaker_recognition"
    )

    # Compute embedding
    with torch.no_grad():
        embedding = verification_model.encode_batch(signal)

    return embedding.squeeze(0).detach()


def diarize_and_identify(audio_path, reference_embedding, threshold=0.65):
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token is None:
        logger.error("HUGGINGFACE_TOKEN environment variable not set.")
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    logger.info("Performing speaker diarization...")
    diarization_result = pipeline(audio_path)
    logger.info("Diarization completed.")

    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/speaker_recognition"
    )

    target_speaker_segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        signal = torch.from_numpy(torch.load(audio_path)).unsqueeze(0)
        with torch.no_grad():
            embedding = verification_model.encode_batch(signal).squeeze(0).detach()
        score = F.cosine_similarity(reference_embedding, embedding, dim=0)
        if score.item() > threshold:
            target_speaker_segments.append(segment)

    logger.info(f"Identified {len(target_speaker_segments)} segments matching the reference speaker.")
    return target_speaker_segments
