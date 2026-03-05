from typing import List
from modules.aadnk import modelCache
from modules.aadnk.config import ModelConfig
from modules.aadnk.whisper.abstractWhisperContainer import AbstractWhisperContainer

def create_whisper_container(whisper_implementation: str, 
                             model_name: str, device: str = None, compute_type: str = "float16",
                             download_root: str = None,
                             cache: modelCache = None, models: List[ModelConfig] = []) -> AbstractWhisperContainer:
    print("Creating whisper container for " + whisper_implementation)

    if (whisper_implementation == "whisper"):
        from modules.aadnk.whisper.whisperContainer import WhisperContainer
        return WhisperContainer(model_name=model_name, device=device, compute_type=compute_type, download_root=download_root, cache=cache, models=models)
    elif (whisper_implementation == "faster-whisper" or whisper_implementation == "faster_whisper"):
        from modules.aadnk.whisper.fasterWhisperContainer import FasterWhisperContainer
        return FasterWhisperContainer(model_name=model_name, device=device, compute_type=compute_type, download_root=download_root, cache=cache, models=models)
    elif (whisper_implementation == "dummy-whisper" or whisper_implementation == "dummy_whisper" or whisper_implementation == "dummy"):
        # This is useful for testing
        from modules.aadnk.whisper.dummyWhisperContainer import DummyWhisperContainer
        return DummyWhisperContainer(model_name=model_name, device=device, compute_type=compute_type, download_root=download_root, cache=cache, models=models)
    else:
        raise ValueError("Unknown Whisper implementation: " + whisper_implementation)