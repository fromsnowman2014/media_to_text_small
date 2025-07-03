"""
Subtitle generation functionality.
"""

import os
import re
from datetime import timedelta
from typing import List, Dict, Optional, Any, Union

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.subtitle")

def format_timestamp(seconds: float, include_millis: bool = True) -> str:
    """
    Format seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        include_millis: Whether to include milliseconds
        
    Returns:
        Formatted timestamp string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if include_millis:
        millis = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{millis:03}"
    else:
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


class SubtitleGenerator:
    """
    Generates subtitle files (SRT, VTT) from transcription segments.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the subtitle generator.
        
        Args:
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = get_logger("media_to_text.subtitle", debug_mode=True)
    
    def segments_to_srt(
        self,
        segments: List[Dict[str, Any]],
        translated_text_key: Optional[str] = None
    ) -> str:
        """
        Convert transcription segments to SRT format.
        
        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text'
            translated_text_key: If provided, use this key for translated text
            
        Returns:
            SRT formatted string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments):
            # Check required fields
            if not all(k in segment for k in ['start', 'end', 'text']):
                logger.warning(f"Segment {i} missing required fields, skipping")
                continue
            
            # Segment index
            srt_lines.append(f"{i+1}")
            
            # Timestamps
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            srt_lines.append(f"{start_time} --> {end_time}")
            
            # Text (use translated text if available and specified)
            if translated_text_key and translated_text_key in segment:
                text = segment[translated_text_key]
            else:
                text = segment['text']
            
            # Clean up the text
            text = text.strip().replace('\n', ' ')
            srt_lines.append(text)
            
            # Empty line between entries
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def segments_to_vtt(
        self,
        segments: List[Dict[str, Any]],
        translated_text_key: Optional[str] = None
    ) -> str:
        """
        Convert transcription segments to WebVTT format.
        
        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text'
            translated_text_key: If provided, use this key for translated text
            
        Returns:
            WebVTT formatted string
        """
        vtt_lines = ["WEBVTT", ""]
        
        for i, segment in enumerate(segments):
            # Check required fields
            if not all(k in segment for k in ['start', 'end', 'text']):
                logger.warning(f"Segment {i} missing required fields, skipping")
                continue
            
            # Segment index (optional in VTT)
            vtt_lines.append(f"{i+1}")
            
            # Timestamps (note VTT uses . instead of , for milliseconds)
            start_time = format_timestamp(segment['start']).replace(',', '.')
            end_time = format_timestamp(segment['end']).replace(',', '.')
            vtt_lines.append(f"{start_time} --> {end_time}")
            
            # Text (use translated text if available and specified)
            if translated_text_key and translated_text_key in segment:
                text = segment[translated_text_key]
            else:
                text = segment['text']
            
            # Clean up the text
            text = text.strip().replace('\n', ' ')
            vtt_lines.append(text)
            
            # Empty line between entries
            vtt_lines.append("")
        
        return "\n".join(vtt_lines)
    
    def generate_subtitle_file(
        self,
        segments: List[Dict[str, Any]],
        output_path: str,
        format_type: str = 'srt',
        translated_text_key: Optional[str] = None
    ) -> str:
        """
        Generate subtitle file from segments.
        
        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text'
            output_path: Path to write the subtitle file
            format_type: Subtitle format ('srt' or 'vtt')
            translated_text_key: If provided, use this key for translated text
            
        Returns:
            Path to the output file
        """
        format_type = format_type.lower()
        if format_type not in ['srt', 'vtt']:
            logger.warning(f"Unsupported subtitle format: {format_type}. Using 'srt' instead.")
            format_type = 'srt'
        
        # Generate subtitle content
        if format_type == 'srt':
            content = self.segments_to_srt(segments, translated_text_key)
        else:
            content = self.segments_to_vtt(segments, translated_text_key)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"{format_type.upper()} subtitle file written to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error writing subtitle file: {str(e)}")
            raise RuntimeError(f"Error writing subtitle file: {str(e)}")
    
    def split_long_segments(
        self, 
        segments: List[Dict[str, Any]], 
        max_chars: int = 40,
        min_duration: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Split long segments into smaller chunks for better subtitles.
        
        Args:
            segments: List of segment dictionaries
            max_chars: Maximum characters per line
            min_duration: Minimum duration for a segment in seconds
            
        Returns:
            List of modified segments with shorter text chunks
        """
        new_segments = []
        
        for segment in segments:
            # Skip segments missing required fields
            if not all(k in segment for k in ['start', 'end', 'text']):
                new_segments.append(segment)
                continue
            
            text = segment['text'].strip()
            duration = segment['end'] - segment['start']
            
            # Only split if segment is long enough and exceeds max chars
            if len(text) <= max_chars or duration < min_duration:
                new_segments.append(segment)
                continue
            
            # Try to split on sentence boundaries first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # If only one sentence, split on comma or other breaks
            if len(sentences) == 1:
                phrases = re.split(r'(?<=[,;:])\s+', text)
                if len(phrases) > 1:
                    sentences = phrases
            
            # If still only one phrase, split by word to fit max_chars
            if len(sentences) == 1:
                words = text.split()
                sentences = []
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        if current:
                            current += " " + word
                        else:
                            current = word
                    else:
                        sentences.append(current)
                        current = word
                if current:
                    sentences.append(current)
            
            # Create new segments
            time_per_char = duration / len(text) if text else min_duration
            current_time = segment['start']
            
            for i, sentence in enumerate(sentences):
                char_count = len(sentence)
                segment_duration = max(min_duration, char_count * time_per_char)
                end_time = min(segment['end'], current_time + segment_duration)
                
                new_segment = {
                    'id': f"{segment.get('id', 0)}.{i}",
                    'start': current_time,
                    'end': end_time,
                    'text': sentence
                }
                
                # Copy additional fields
                for key, value in segment.items():
                    if key not in ['id', 'start', 'end', 'text']:
                        new_segment[key] = value
                
                new_segments.append(new_segment)
                current_time = end_time
        
        return new_segments
