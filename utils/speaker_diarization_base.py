print("Importing speaker_diarization_base.py")
from abc import ABC, abstractmethod
print("Imported abc module successfully")

class SpeakerDiarizationBase(ABC):
    print("Defining SpeakerDiarizationBase")

    @abstractmethod
    def load_pipeline(self):
        pass

    @abstractmethod
    def diarize(self, audio_path):
        pass

    @abstractmethod
    def identify_speakers(self, audio_path, diarization_result, reference_embedding):
        pass
print("SpeakerDiarizationBase defined successfully")
