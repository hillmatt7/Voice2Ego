import os
import torch
import logging
import torch.nn.functional as F
import warnings
from pyannote.audio import Pipeline
import librosa

from utils.speaker_diarization_base import SpeakerDiarizationBase

# Configure logging
logger = logging.getLogger(__name__)

class SpeakerDiarizationWrapper(SpeakerDiarizationBase):
    def __init__(self, hf_token, threshold, verification_model):
        self.hf_token = hf_token
        self.threshold = threshold
        self.pipeline = None
        self.verification = verification_model  # Use the passed verification model

    def load_pipeline(self):
        # Load the pyannote.audio pipeline
        logger.info("Loading pyannote.audio pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=self.hf_token
        )

    def diarize(self, audio_path):
        # Perform diarization
        logger.info("Performing speaker diarization...")
        diarization_result = self.pipeline(audio_path)
        logger.info("Speaker diarization completed.")
        return diarization_result

    def identify_speakers(self, audio_path, diarization_result, reference_embedding):
        # Identify segments matching the reference embedding
        logger.info("Identifying target speaker segments...")
        target_speaker_segments = []

        for segment, _, _ in diarization_result.itertracks(yield_label=True):
            logger.debug(f"Processing segment: {segment}")

            # Load segment audio
            signal, fs = librosa.load(
                audio_path, sr=16000, offset=segment.start, duration=segment.duration
            )
            if len(signal) == 0:
                continue  # Skip empty segments

            signal = torch.from_numpy(signal).unsqueeze(0)
            # Compute embedding for segment
            with torch.no_grad():
                embedding = self.verification.encode_batch(signal).squeeze(0).detach()
            # Compute similarity score using cosine similarity
            score = F.cosine_similarity(reference_embedding, embedding, dim=0)
            # Determine if it's the target speaker
            if score.item() > self.threshold:
                logger.debug(f"Segment {segment} matches target speaker with score {score.item()}.")
                target_speaker_segments.append(segment)
            else:
                logger.debug(f"Segment {segment} does not match target speaker (score: {score.item()}).")

        logger.info(f"Identified {len(target_speaker_segments)} segments matching the reference speaker.")
        return target_speaker_segments
