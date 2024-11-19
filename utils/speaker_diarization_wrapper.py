# utils/speaker_diarization_wrapper.py

import os
import torch
import logging
import torch.nn.functional as F
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.utils.signal import binarize
from pyannote.core import SlidingWindowFeature, Annotation
import librosa

# Configure logging
logger = logging.getLogger(__name__)

class SpeakerDiarizationWrapper(SpeakerDiarization):
    def __init__(self, hf_token):
        super().__init__()
        self.hf_token = hf_token

    @classmethod
    def from_pretrained(cls, model_name_or_path, use_auth_token=None):
        # Load the base pipeline
        base_pipeline = super(SpeakerDiarizationWrapper, cls).from_pretrained(
            model_name_or_path,
            use_auth_token=use_auth_token
        )
        # Create an instance of the custom class
        custom_pipeline = cls(use_auth_token)
        # Copy attributes from the base pipeline
        custom_pipeline.__dict__.update(base_pipeline.__dict__)
        return custom_pipeline


def get_embeddings(self, file, segmentation=None, exclude_overlap=False, **kwargs):
    if segmentation is None:
        segmentation = self.get_segmentation(file)
    
    # Binarize the segmentation to get hard labels
    hard_segmentation = binarize(segmentation)
    
    # Get the timeline of speech segments
    timeline = hard_segmentation.get_timeline()

    # Exclude overlapping segments if exclude_overlap is True
    if exclude_overlap:
        overlaps = timeline.get_overlap()
        timeline = timeline.extrude(overlaps)

    # Extract waveforms
    waveforms = []
    for segment in timeline:
        waveform, _ = self.audio.crop(file, segment, mode="strict")
        waveforms.append(waveform)

    # Handle empty waveforms
    waveforms = [w for w in waveforms if w.numel() > 0]
    if not waveforms:
        return torch.empty(0)

    # Find the maximum length
    max_length = max([w.shape[1] for w in waveforms])

    # Pad waveforms to the maximum length
    padded_waveforms = []
    for waveform in waveforms:
        length = waveform.shape[1]
        if length < max_length:
            pad_size = max_length - length
            # Pad with zeros at the end
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            padded_waveforms.append(padded_waveform)
        else:
            padded_waveforms.append(waveform)

    # Stack the padded waveforms
    waveform_batch = torch.vstack(padded_waveforms)

    # Extract embeddings
    embeddings = self.embedding(waveform_batch)
    return embeddings
