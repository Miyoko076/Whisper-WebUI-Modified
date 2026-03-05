from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.aadnk.vad import VadSileroTranscription
from modules.aadnk.whisper.abstractWhisperContainer import AbstractWhisperCallback
from modules.whisper.data_classes import TranscriptionPipelineParams, Segment
import gradio as gr
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
import time
import torch

class AadnkVadWrapper(BaseTranscriptionPipeline):
    def __init__(self, wrapped_pipeline: BaseTranscriptionPipeline):
        # This wrapper inherits from BaseTranscriptionPipeline, so we need to initialize the parent
        super().__init__(model_dir=wrapped_pipeline.model_dir, 
                         diarization_model_dir=wrapped_pipeline.diarizer.model_dir, 
                         uvr_model_dir=wrapped_pipeline.music_separator.model_dir, 
                         output_dir=wrapped_pipeline.output_dir)

        # Now, replace the inherited instances with the ones from the wrapped pipeline
        self.wrapped_pipeline = wrapped_pipeline
        self.diarizer = wrapped_pipeline.diarizer
        self.vad = wrapped_pipeline.vad
        self.music_separator = wrapped_pipeline.music_separator
        self.model = wrapped_pipeline.model
        self.current_model_size = wrapped_pipeline.current_model_size
        
        # Init the aadnk VAD model
        self.aadnk_vad_model = VadSileroTranscription()

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ):
        return self.wrapped_pipeline.transcribe(audio, progress, progress_callback, *whisper_params)

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        return self.wrapped_pipeline.update_model(model_size, compute_type, progress)

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        
        all_params = list(pipeline_params)
        aadnk_vad_enable = all_params[-2]
        aadnk_vad_mode = all_params[-1]
        original_pipeline_params = tuple(all_params[:-2])

        if not aadnk_vad_enable:
            # If disabled, just call the original run method
            return self.wrapped_pipeline.run(audio, progress, file_format, add_timestamp, progress_callback, *original_pipeline_params)
        else:
            # If enabled, execute the aadnk VAD pipeline
            start_time = time.time()
            
            params = TranscriptionPipelineParams.from_list(list(original_pipeline_params))
            whisper_params_list = params.whisper.to_list()

            # The audio needs to be a file path for aadnk's VAD
            if not isinstance(audio, str):
                 # This is a limitation for now. The aadnk code expects a file path.
                 # The original pipeline can handle numpy arrays. We'd need to save the array to a temp file.
                raise TypeError("aadnk VAD pipeline currently only supports file paths as audio input.")

            # 1. Create a callback for the aadnk VAD pipeline to call for each chunk
            class WrapperCallback(AbstractWhisperCallback):
                def __init__(self, pipeline, whisper_params, progress_obj):
                    self.pipeline = pipeline
                    self.whisper_params = whisper_params
                    self.progress = progress_obj

                def invoke(self, audio_chunk, segment_index, prompt, lang, progress_listener):
                    # The `transcribe` method of the wrapped pipeline handles the core whisper logic.
                    # We need to call it with the audio chunk and the correct parameters.
                    # Note: The progress_listener from aadnk is not directly compatible with gradio progress,
                    # so we'll just pass the main progress object for now.
                    segments, _ = self.pipeline.transcribe(audio_chunk, self.progress, None, *self.whisper_params)
                    # The aadnk pipeline expects a dictionary with a specific structure.
                    return {"segments": [s.to_dict() for s in segments]}
            
            whisper_callback = WrapperCallback(self.wrapped_pipeline, whisper_params_list, progress)

            # 2. Set up the aadnk VAD transcription config
            from modules.aadnk.vad import TranscriptionConfig, NonSpeechStrategy
            
            vad_strategy = NonSpeechStrategy.SKIP
            if aadnk_vad_mode == "silero-vad":
                vad_strategy = NonSpeechStrategy.CREATE_SEGMENT
            elif aadnk_vad_mode == "silero-vad-expand-into-gaps":
                vad_strategy = NonSpeechStrategy.EXPAND_SEGMENT

            # These are default values from aadnk/app.py, we can make them configurable later
            config = TranscriptionConfig(
                non_speech_strategy=vad_strategy,
                max_silent_period=5.0,
                max_merge_size=30.0
            )

            # 3. Run the transcription
            progress(0, desc="[aadnk VAD] Running VAD and transcription...")
            result_dict = self.aadnk_vad_model.transcribe(audio, whisper_callback, config)

            # 4. Convert result back to Segment objects
            segments_result = [Segment.from_dict(s) for s in result_dict.get("segments", [])]

            total_elapsed_time = time.time() - start_time
            progress(1.0, desc="Finished.")
            return segments_result, total_elapsed_time

