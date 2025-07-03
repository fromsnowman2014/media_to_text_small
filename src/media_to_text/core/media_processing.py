"""
Media processing utilities for audio extraction and format conversion.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import av  # PyAV for media handling

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.media_processing")

def extract_audio_from_video(
    video_path: str, 
    output_path: Optional[str] = None,
    audio_format: str = 'wav',
    audio_codec: str = 'pcm_s16le',
    sample_rate: int = 16000
) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the input video file
        output_path: Path for the output audio file, generated if None
        audio_format: Format for the output audio file
        audio_codec: Codec for the output audio file
        sample_rate: Sample rate for the output audio
        
    Returns:
        Path to the extracted audio file
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path if not provided
    if output_path is None:
        video_filename = Path(video_path).stem
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"{video_filename}.{audio_format}")
    
    try:
        logger.info(f"Extracting audio from video: {video_path}")
        
        # Open input container
        input_container = av.open(video_path)
        
        # Find audio stream
        audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
        if audio_stream is None:
            logger.error(f"No audio stream found in {video_path}")
            raise ValueError(f"No audio stream found in {video_path}")
        
        # Open output container
        output_container = av.open(output_path, mode='w')
        
        # Add output audio stream
        output_stream = output_container.add_stream(audio_codec)
        output_stream.sample_rate = sample_rate
        
        # Convert audio
        for frame in input_container.decode(audio_stream):
            # Resample frame if needed
            if frame.sample_rate != sample_rate:
                frame = frame.resample(sample_rate)
            
            # Encode frame and write to output
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        
        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        # Close containers
        output_container.close()
        input_container.close()
        
        logger.info(f"Audio extracted to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise RuntimeError(f"Error extracting audio: {str(e)}")


def convert_audio_format(
    input_path: str,
    output_format: str = 'wav',
    output_path: Optional[str] = None,
    audio_codec: Optional[str] = None,
    sample_rate: int = 16000
) -> str:
    """
    Convert audio to a different format.
    
    Args:
        input_path: Path to the input audio file
        output_format: Format for the output audio
        output_path: Path for the output audio file, generated if None
        audio_codec: Codec for the output audio, defaults to appropriate codec for format
        sample_rate: Sample rate for the output audio
        
    Returns:
        Path to the converted audio file
    """
    if not os.path.exists(input_path):
        logger.error(f"Audio file not found: {input_path}")
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    # Set default codec based on output format if not specified
    if audio_codec is None:
        if output_format == 'wav':
            audio_codec = 'pcm_s16le'
        elif output_format == 'mp3':
            audio_codec = 'libmp3lame'
        elif output_format == 'flac':
            audio_codec = 'flac'
        else:
            audio_codec = 'aac'  # Default for other formats
    
    # Generate output path if not provided
    if output_path is None:
        audio_filename = Path(input_path).stem
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"{audio_filename}.{output_format}")
    
    try:
        logger.info(f"Converting audio {input_path} to {output_format}")
        
        # Open input container
        input_container = av.open(input_path)
        
        # Find audio stream
        audio_stream = next((s for s in input_container.streams if s.type == 'audio'), None)
        if audio_stream is None:
            logger.error(f"No audio stream found in {input_path}")
            raise ValueError(f"No audio stream found in {input_path}")
        
        # Open output container
        output_container = av.open(output_path, mode='w')
        
        # Add output audio stream
        output_stream = output_container.add_stream(audio_codec)
        output_stream.sample_rate = sample_rate
        
        # Convert audio
        for frame in input_container.decode(audio_stream):
            # Resample frame if needed
            if frame.sample_rate != sample_rate:
                frame = frame.resample(sample_rate)
            
            # Encode frame and write to output
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        
        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        # Close containers
        output_container.close()
        input_container.close()
        
        logger.info(f"Audio converted to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise RuntimeError(f"Error converting audio: {str(e)}")
