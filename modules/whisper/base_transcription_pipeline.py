import os
import whisper
import ctranslate2
import gradio as gr
import torchaudio
from abc import ABC, abstractmethod
from typing import BinaryIO, Union, Tuple, List, Callable
import numpy as np
from datetime import datetime
from faster_whisper.vad import VadOptions
import gc
from copy import deepcopy
import time

from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR)
from modules.utils.constants import *
from modules.utils.logger import get_logger
from modules.utils.subtitle_manager import *
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, load_yaml, save_yaml, read_file
from modules.utils.audio_manager import validate_audio
from modules.whisper.data_classes import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD
from modules.aadnk.vad import VadSileroTranscription
from modules.aadnk.whisper.abstractWhisperContainer import AbstractWhisperCallback
from modules.aadnk.vad import TranscriptionConfig as AadnkTranscriptionConfig, NonSpeechStrategy
import tempfile



logger = get_logger()


class BaseTranscriptionPipeline(ABC):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.diarizer = Diarizer(
            model_dir=diarization_model_dir
        )
        self.vad = SileroVAD()
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR")
        )

        self.model = None
        self.current_model_size = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    @abstractmethod
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ):
        """Inference whisper model to transcribe"""
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """Initialize whisper model"""
        pass

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        
        all_params = list(pipeline_params)
        
        # --- 1. AADNK 및 할루시네이션 파라미터 15개 추출 ---
        aadnk_params = all_params[-15:]
        aadnk_vad_enable = aadnk_params[0]
        aadnk_vad_mode = aadnk_params[1]
        aadnk_vad_merge_window = aadnk_params[2]
        aadnk_vad_max_merge_size = aadnk_params[3]
        aadnk_vad_padding = aadnk_params[4]
        aadnk_vad_prompt_window = aadnk_params[5]
        
        hal_enable = aadnk_params[6]
        hal_cr_threshold = aadnk_params[7]
        hal_strategy = aadnk_params[8]
        hal_temp_start = aadnk_params[9]
        hal_temp_step = aadnk_params[10]
        hal_temp_retry = aadnk_params[11]
        hal_time_start = aadnk_params[12]
        hal_time_step = aadnk_params[13]
        hal_time_retry = aadnk_params[14]
        
        original_pipeline_params = tuple(all_params[:-15])

        if not validate_audio(audio):
            return [Segment()], 0

        params = TranscriptionPipelineParams.from_list(list(original_pipeline_params))
        params = self.validate_gradio_values(params)
        bgm_params = params.bgm_separation
        vad_params = params.vad

        if aadnk_vad_enable and vad_params.vad_filter:
            logger.info(f"Both VADs enabled. Overriding existing VAD with 'aadnk' VAD.")

        if bgm_params.is_separate_bgm:
            music, audio, _ = self.music_separator.separate(
                audio=audio, model_name=bgm_params.uvr_model_size, device=bgm_params.uvr_device,
                segment_size=bgm_params.segment_size, save_file=bgm_params.save_file, progress=progress
            )

            if audio.ndim >= 2:
                audio = audio.mean(axis=1)
                if self.music_separator.audio_info is None:
                    origin_sample_rate = 16000
                else:
                    origin_sample_rate = self.music_separator.audio_info.sample_rate
                audio = self.resample_audio(audio=audio, original_sample_rate=origin_sample_rate)

            if bgm_params.enable_offload: self.music_separator.offload()

        if not aadnk_vad_enable:
            # --- 기존 로직 ---
            start_time = time.time()
            whisper_params, diarization_params = params.whisper, params.diarization
            origin_audio = deepcopy(audio)

            if vad_params.vad_filter:
                progress(0, desc="Filtering silent parts from audio..")
                vad_options = VadOptions(
                    threshold=vad_params.threshold, min_speech_duration_ms=vad_params.min_speech_duration_ms,
                    max_speech_duration_s=vad_params.max_speech_duration_s, min_silence_duration_ms=vad_params.min_silence_duration_ms,
                    speech_pad_ms=vad_params.speech_pad_ms
                )
                vad_processed, speech_chunks = self.vad.run(audio=audio, vad_parameters=vad_options, progress=progress)
                if vad_processed.size > 0:
                    audio = vad_processed
                else:
                    vad_params.vad_filter = False

            result, info = self.transcribe(audio, progress, progress_callback, *whisper_params.to_list())
            if whisper_params.enable_offload: self.offload()

            if vad_params.vad_filter:
                restored_result = self.vad.restore_speech_timestamps(segments=result, speech_chunks=speech_chunks)
                if restored_result:
                    result = restored_result
                else:
                    logger.info("VAD detected no speech segments in the audio.")

            if diarization_params.is_diarize:
                progress(0.99, desc="Diarizing speakers..")
                result, elapsed_time_diarization = self.diarizer.run(
                    audio=origin_audio, use_auth_token=diarization_params.hf_token if diarization_params.hf_token else os.environ.get("HF_TOKEN"),
                    transcribed_result=result, device=diarization_params.diarization_device
                )
                if diarization_params.enable_offload: self.diarizer.offload()

            self.cache_parameters(params=params, file_format=file_format, add_timestamp=add_timestamp)

            if not result:
                logger.info(f"Whisper did not detected any speech segments in the audio.")
                result = [Segment()]

            progress(1.0, desc="Finished.")
            total_elapsed_time = time.time() - start_time
            return result, total_elapsed_time
            
        else:
            # --- 2. AADNK VAD 로직 (할루시네이션 복구 포함) ---
            start_time = time.time()
            whisper_params = params.whisper
            diarization_params = params.diarization

            audio_path = audio
            if not isinstance(audio, str):
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_path = temp_audio_file.name
                torchaudio.save(audio_path, torch.from_numpy(audio).unsqueeze(0), 16000)

            hal_configs = {
                "enable": hal_enable,
                "cr_threshold": hal_cr_threshold,
                "strategy": hal_strategy,
                "temp_start": hal_temp_start,
                "temp_step": hal_temp_step,
                "temp_retry": int(hal_temp_retry),
                "time_start_sec": hal_time_start,
                "time_step_sec": hal_time_step,
                "time_retry": int(hal_time_retry)
            }

            class WrapperCallback(AbstractWhisperCallback):
                def __init__(self, pipeline, base_whisper_params, progress_obj, configs):
                    self.pipeline = pipeline
                    self.base_whisper_params = base_whisper_params
                    self.progress = progress_obj
                    self.configs = configs

                def invoke(self, audio_chunk, segment_index, prompt, lang, progress_listener):
                    lang_code = self.base_whisper_params.lang or lang

                    # 결과 포맷팅 도우미 함수 (위치 최상단으로 이동)
                    def format_result(segs, inf):
                        lc = lang_code
                        if inf:
                            if hasattr(inf, 'language'): lc = inf.language
                            elif isinstance(inf, dict) and 'language' in inf: lc = inf.get('language')
                        segs = segs or []
                        txt = " ".join([s.text for s in segs])
                        return {"text": txt, "segments": [s.model_dump() for s in segs], "language": lc}

                    # 할루시네이션 필터 미사용 시 기존 로직대로 1회만 실행하고 즉시 반환
                    if not self.configs["enable"]:
                        segments, info = self.pipeline.transcribe(audio_chunk, self.progress, None, *self.base_whisper_params.to_list())
                        return format_result(segments, info)

                    # --- 할루시네이션 순차적 복구 알고리즘 시작 ---
                    best_segments = None
                    best_info = None
                    best_logprob = -float('inf')

                    def attempt_transcription(audio_data, temp_val):
                        nonlocal best_segments, best_info, best_logprob
                        current_params = deepcopy(self.base_whisper_params)
                        current_params.temperature = temp_val
                        
                        segments, info = self.pipeline.transcribe(audio_data, self.progress, None, *current_params.to_list())
                        
                        is_hallucinating = False
                        current_avg_logprob = -float('inf')
                        
                        if segments:
                            logprobs = []
                            for s in segments:
                                cr = getattr(s, 'compression_ratio', 0)
                                lp = getattr(s, 'avg_logprob', -1.0)
                                logprobs.append(lp)
                                if cr is not None and cr >= self.configs["cr_threshold"]:
                                    is_hallucinating = True
                            current_avg_logprob = sum(logprobs) / len(logprobs)
                        else:
                            is_hallucinating = True

                        if current_avg_logprob > best_logprob:
                            best_logprob = current_avg_logprob
                            best_segments = segments
                            best_info = info

                        return not is_hallucinating, segments, info

                    # 1. Baseline: 최초 1회 기본 시도
                    success, segs, info = attempt_transcription(audio_chunk, self.configs["temp_start"])
                    if success:
                        logger.info(f"[Seg {segment_index}] 정상 통과 (Temp: {self.configs['temp_start']:.2f})")
                        return format_result(segs, info)

                    # 2. Phase 1: Temp+ 전략 수행
                    if "Temp+" in self.configs["strategy"]:
                        for retry in range(1, self.configs["temp_retry"] + 1):
                            current_temp = round(min(self.configs["temp_start"] + (self.configs["temp_step"] * retry), 1.0), 2)
                            logger.info(f"[Seg {segment_index}] 환각 감지. 온도를 {current_temp:.2f}로 변경하여 재시도합니다. ({retry}/{self.configs['temp_retry']})")
                            success, segs, info = attempt_transcription(audio_chunk, current_temp)
                            if success:
                                return format_result(segs, info)

                    # 3. Phase 2: Time+ 전략 수행 (온도 조절 실패 시)
                    if "Time+" in self.configs["strategy"]:
                        for retry in range(1, self.configs["time_retry"] + 1):
                            trunc_sec = self.configs["time_start_sec"] + (self.configs["time_step_sec"] * (retry - 1))
                            trunc_samples = int(trunc_sec * 16000)
                            
                            if trunc_samples >= len(audio_chunk):
                                logger.warning(f"[Seg {segment_index}] 절삭 시간이 오디오 길이를 초과했습니다. Time+ 중단.")
                                break
                                
                            logger.info(f"[Seg {segment_index}] 앞 {trunc_sec:.2f}s 절삭 후 기본 온도로 재시도합니다. ({retry}/{self.configs['time_retry']})")
                            truncated_audio = audio_chunk[trunc_samples:]
                            
                            success, segs, info = attempt_transcription(truncated_audio, self.configs["temp_start"])
                            if success:
                                return format_result(segs, info)

                    # 4. 모든 복구 실패: 베스트 반환
                    logger.warning(f"[Seg {segment_index}] 모든 복구 전략 실패. 가장 품질이 높았던 결과를 강제로 반환합니다.")
                    return format_result(best_segments, best_info)

            whisper_callback = WrapperCallback(self, whisper_params, progress, hal_configs)

            aadnk_vad_model = VadSileroTranscription()
            
            vad_strategy = NonSpeechStrategy.SKIP
            if aadnk_vad_mode == "silero-vad":
                vad_strategy = NonSpeechStrategy.CREATE_SEGMENT
            elif aadnk_vad_mode == "silero-vad-skip-gaps":
                vad_strategy = NonSpeechStrategy.SKIP
            elif aadnk_vad_mode == "silero-vad-expand-into-gaps":
                vad_strategy = NonSpeechStrategy.EXPAND_SEGMENT

            config = AadnkTranscriptionConfig(
                non_speech_strategy=vad_strategy,
                max_silent_period=aadnk_vad_merge_window,
                max_merge_size=aadnk_vad_max_merge_size
            )

            progress(0, desc="[aadnk VAD] Running VAD and transcription...")
            result_dict = aadnk_vad_model.transcribe(audio_path, whisper_callback, config)
            segments_result = [Segment.model_validate(s) for s in result_dict.get("segments", [])]

            if diarization_params.is_diarize:
                progress(0.99, desc="Diarizing speakers..")
                origin_audio, _ = torchaudio.load(audio_path)
                segments_result, _ = self.diarizer.run(
                    audio=origin_audio, use_auth_token=diarization_params.hf_token if diarization_params.hf_token else os.environ.get("HF_TOKEN"),
                    transcribed_result=segments_result, device=diarization_params.diarization_device
                )
                if diarization_params.enable_offload: self.diarizer.offload()

            if not segments_result:
                logger.info(f"Whisper did not detected any speech segments in the audio.")
                segments_result = [Segment()]

            total_elapsed_time = time.time() - start_time
            progress(1.0, desc="Finished.")
            return segments_result, total_elapsed_time

    def transcribe_file(self,
                        files: Optional[List] = None,
                        input_folder_path: Optional[str] = None,
                        include_subdirectory: Optional[str] = None,
                        save_same_dir: Optional[str] = None,
                        file_format: str = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        *pipeline_params,
                        ) -> Tuple[str, List]:
        """
        Write subtitle file from Files

        Parameters
        ----------
        files: list
            List of files to transcribe from gr.Files()
        input_folder_path: Optional[str]
            Input folder path to transcribe from gr.Textbox(). If this is provided, `files` will be ignored and
            this will be used instead.
        include_subdirectory: Optional[str]
            When using `input_folder_path`, whether to include all files in the subdirectory or not
        save_same_dir: Optional[str]
            When using `input_folder_path`, whether to save output in the same directory as inputs or not, in addition
            to the original output directory. This feature is only available when using `input_folder_path`, because
            gradio only allows to use cached file path in the function yet.
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            if input_folder_path:
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)
            if isinstance(files, str):
                files = [files]
            if files and isinstance(files[0], gr.utils.NamedString):
                files = [file.name for file in files]

            files_info = {}
            for file in files:
                transcribed_segments, time_for_task = self.run(
                    file,
                    progress,
                    file_format,
                    add_timestamp,
                    None,
                    *pipeline_params,
                )

                file_name, file_ext = os.path.splitext(os.path.basename(file))
                if save_same_dir and input_folder_path:
                    output_dir = os.path.dirname(file)
                    subtitle, file_path = generate_file(
                        output_dir=output_dir,
                        output_file_name=file_name,
                        output_format=file_format,
                        result=transcribed_segments,
                        add_timestamp=add_timestamp,
                        **writer_options
                    )

                subtitle, file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=file_name,
                    output_format=file_format,
                    result=transcribed_segments,
                    add_timestamp=add_timestamp,
                    **writer_options
                )
                files_info[file_name] = {"subtitle": read_file(file_path), "time_for_task": time_for_task, "path": file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            result_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            result_file_path = [info['path'] for info in files_info.values()]

            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing file: {e}") from e

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: str = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = "Mic"
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return result_str, file_path
        except Exception as e:
            raise RuntimeError(f"Error transcribing mic: {e}") from e

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, str]:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtube_link: str
            URL of the Youtube video to transcribe from gr.Textbox()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"

            if os.path.exists(audio):
                os.remove(audio)

            return result_str, file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing youtube: {e}") from e

    def get_compute_type(self):
        if "float16" in self.available_compute_types:
            return "float16"
        if "float32" in self.available_compute_types:
            return "float32"
        else:
            return self.available_compute_types[0]

    def get_available_compute_type(self):
        if self.device == "cuda":
            return list(ctranslate2.get_supported_compute_types("cuda"))
        else:
            return list(ctranslate2.get_supported_compute_types("cpu"))

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        """
        Get {hours} {minutes} {seconds} time format string

        Parameters
        ----------
        elapsed_time: str
            Elapsed time for transcription

        Returns
        ----------
        Time format string
        """
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            if not BaseTranscriptionPipeline.is_sparse_api_supported():
                # Device `SparseMPS` is not supported for now. See : https://github.com/pytorch/pytorch/issues/87886
                return "cpu"
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_sparse_api_supported():
        if not torch.backends.mps.is_available():
            return False

        try:
            device = torch.device("mps")
            sparse_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1], [2, 3]]),
                values=torch.tensor([1, 2]),
                size=(4, 4),
                device=device
            )
            return True
        except RuntimeError:
            return False

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        """Remove gradio cached files"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    @staticmethod
    def validate_gradio_values(params: TranscriptionPipelineParams):
        """
        Validate gradio specific values that can't be displayed as None in the UI.
        Related issue : https://github.com/gradio-app/gradio/issues/8723
        """
        if params.whisper.lang is None:
            pass
        elif params.whisper.lang == AUTOMATIC_DETECTION:
            params.whisper.lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            params.whisper.lang = language_code_dict[params.whisper.lang]

        if params.whisper.initial_prompt == GRADIO_NONE_STR:
            params.whisper.initial_prompt = None
        if params.whisper.prefix == GRADIO_NONE_STR:
            params.whisper.prefix = None
        if params.whisper.hotwords == GRADIO_NONE_STR:
            params.whisper.hotwords = None
        if params.whisper.max_new_tokens == GRADIO_NONE_NUMBER_MIN:
            params.whisper.max_new_tokens = None
        if params.whisper.hallucination_silence_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.hallucination_silence_threshold = None
        if params.whisper.language_detection_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.language_detection_threshold = None
        if params.vad.max_speech_duration_s == GRADIO_NONE_NUMBER_MAX:
            params.vad.max_speech_duration_s = float('inf')
        return params

    @staticmethod
    def cache_parameters(
        params: TranscriptionPipelineParams,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ):
        """Cache parameters to the yaml file"""
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        param_to_cache = params.to_dict()

        cached_yaml = {**cached_params, **param_to_cache}
        cached_yaml["whisper"]["add_timestamp"] = add_timestamp
        cached_yaml["whisper"]["file_format"] = file_format

        supress_token = cached_yaml["whisper"].get("suppress_tokens", None)
        if supress_token and isinstance(supress_token, list):
            cached_yaml["whisper"]["suppress_tokens"] = str(supress_token)

        if cached_yaml["whisper"].get("lang", None) is None:
            cached_yaml["whisper"]["lang"] = AUTOMATIC_DETECTION.unwrap()
        else:
            language_dict = whisper.tokenizer.LANGUAGES
            cached_yaml["whisper"]["lang"] = language_dict[cached_yaml["whisper"]["lang"]]

        if cached_yaml["vad"].get("max_speech_duration_s", float('inf')) == float('inf'):
            cached_yaml["vad"]["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX

        if cached_yaml is not None and cached_yaml:
            save_yaml(cached_yaml, DEFAULT_PARAMETERS_CONFIG_PATH)

    @staticmethod
    def resample_audio(audio: Union[str, np.ndarray],
                       new_sample_rate: int = 16000,
                       original_sample_rate: Optional[int] = None,) -> np.ndarray:
        """Resamples audio to 16k sample rate, standard on Whisper model"""
        if isinstance(audio, str):
            audio, original_sample_rate = torchaudio.load(audio)
        else:
            if original_sample_rate is None:
                raise ValueError("original_sample_rate must be provided when audio is numpy array.")
            audio = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        resampled_audio = resampler(audio).numpy()
        return resampled_audio
