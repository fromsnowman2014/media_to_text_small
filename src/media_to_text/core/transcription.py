"""
Core transcription functionality using faster-whisper.
"""

import os
import tempfile
import logging
from typing import Dict, Optional, List, Tuple, Union, Literal
from pathlib import Path

from faster_whisper import WhisperModel

from ..utils.logging_utils import get_logger

# Define whisper model sizes
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large']
SegmentData = Dict[str, Union[int, float, str]]

logger = get_logger("media_to_text.transcription")

class Transcriber:
    """
    Handles audio transcription using the faster-whisper model.
    """
    def __init__(
        self, 
        model_size: str = 'base',
        language: Optional[str] = None, 
        device: str = 'cpu',
        compute_type: str = 'int8',
        debug: bool = False
    ):
        """
        Initialize the transcriber with the specified model and settings.
        
        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code (e.g., 'en', 'ko') or None for auto-detection
            device: Device to use for inference ('cpu' or 'cuda')
            compute_type: Compute type for the model ('int8', 'int16', 'float16', 'float32')
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = get_logger("media_to_text.transcription", debug_mode=True)
        
        if model_size not in MODEL_SIZES:
            logger.warning(f"Invalid model size: {model_size}. Using 'base' instead.")
            model_size = 'base'
        
        logger.info(f"Initializing Whisper model: size={model_size}, device={device}, compute_type={compute_type}")
        
        # Set OpenMP environment variable to prevent conflicts with PyTorch
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
            
        self.language = language
    
    def transcribe(
        self, 
        audio_path: str, 
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True
    ) -> Tuple[List[SegmentData], Dict]:
        """
        Transcribe audio file to text using faster-whisper.
        
        Args:
            audio_path: Path to the audio file
            beam_size: Beam size for decoding
            vad_filter: Whether to use voice activity detection
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            A tuple of (segments, info) where segments is a list of transcribed segments
            and info contains metadata about the transcription
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=beam_size,
                language=self.language,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps
            )
            
            # Convert generator to a list of segment dictionaries for easier processing
            segments_list = []
            for segment in segments:
                segment_dict = {
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                }
                
                if word_timestamps and segment.words:
                    segment_dict['words'] = [
                        {'start': word.start, 'end': word.end, 'word': word.word}
                        for word in segment.words
                    ]
                
                segments_list.append(segment_dict)
            
            logger.info(f"Transcription completed. Detected language: {info.language}")
            return segments_list, info
        
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
    
    def transcribe_to_file(
        self, 
        audio_path: str, 
        output_path: str,
        **kwargs
    ) -> str:
        """
        Transcribe audio and write the result to a file.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to write the transcription
            **kwargs: Additional arguments for the transcribe method
            
        Returns:
            Path to the output file
        """
        segments, info = self.transcribe(audio_path, **kwargs)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n")
        
        logger.info(f"Transcription written to {output_path}")
        return output_path
    
    def transcribe_with_timestamps(
        self, 
        audio_path: str
    ) -> List[Dict]:
        """
        Transcribe audio with detailed timestamps for subtitle generation.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of segments with start time, end time, and text
        """
        segments, _ = self.transcribe(audio_path, word_timestamps=True)
        return segments
