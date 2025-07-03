"""
Utilities for file operations and type detection.
"""

import os
import logging
import mimetypes
from pathlib import Path
from typing import Optional, List, Tuple, Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("media_to_text.file_utils")

# Define supported file types
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi']
SUPPORTED_TEXT_FORMATS = ['.txt', '.srt']
SUPPORTED_DOCUMENT_FORMATS = ['.pdf']

InputType = Literal['audio', 'video', 'text', 'pdf', 'unknown']

def detect_file_type(file_path: str) -> InputType:
    """
    Detect the type of the input file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        A string representing the file type: 'audio', 'video', 'text', 'pdf', or 'unknown'
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension in SUPPORTED_AUDIO_FORMATS:
        return 'audio'
    elif file_extension in SUPPORTED_VIDEO_FORMATS:
        return 'video'
    elif file_extension in SUPPORTED_TEXT_FORMATS:
        return 'text'
    elif file_extension in SUPPORTED_DOCUMENT_FORMATS:
        return 'pdf'
    else:
        # Try to use mimetypes for better detection
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type.startswith('video/'):
                return 'video'
            elif mime_type.startswith('text/') or mime_type == 'application/x-subrip':
                return 'text'
            elif mime_type == 'application/pdf':
                return 'pdf'
    
    logger.warning(f"Unknown file type for: {file_path}")
    return 'unknown'


def get_output_path(input_file: str, output_dir: str, suffix: str, extension: str) -> str:
    """
    Generate the output file path based on the input file and desired suffix.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory for output files
        suffix: Suffix to add to the filename
        extension: File extension for the output file
        
    Returns:
        Path to the output file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name without extension
    base_name = Path(input_file).stem
    output_filename = f"{base_name}{suffix}.{extension.lstrip('.')}"
    
    return os.path.join(output_dir, output_filename)


def find_supported_files(directory: str) -> List[Tuple[str, InputType]]:
    """
    Find all supported files in a directory.
    
    Args:
        directory: Directory path to search for supported files
        
    Returns:
        List of tuples containing (file_path, file_type)
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        raise NotADirectoryError(f"Directory not found: {directory}")
    
    supported_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_type = detect_file_type(file_path)
                if file_type != 'unknown':
                    supported_files.append((file_path, file_type))
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
    
    return supported_files
