"""
Simple text processing utility as a fallback when ML models cannot be loaded.
"""

import os
import re
from typing import Dict, List, Any, Optional

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.simple_text_processor")

class SimpleTextProcessor:
    """
    Provides basic text processing functions as a fallback when ML models cannot be loaded.
    """
    
    @staticmethod
    def simple_summarize(text: str, max_length: int = 150) -> str:
        """
        Create a simple summary by extracting key sentences from the text.
        
        Args:
            text: The text to summarize
            max_length: Maximum word count for the summary
            
        Returns:
            Simple summary of the text
        """
        logger.info("Using simple summarization (extraction-based)")
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return ""
        
        # Score sentences based on position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Score based on position (first and last sentences often contain key information)
            position_score = 1.0 if i < len(sentences) // 4 or i > 3 * len(sentences) // 4 else 0.5
            
            # Score based on sentence length (medium length sentences often contain key information)
            words = sentence.split()
            length = len(words)
            length_score = 0.5 if 5 <= length <= 25 else 0.0
            
            # Score based on keyword presence (placeholder - would need domain keywords)
            keyword_score = 0.0
            
            total_score = position_score + length_score + keyword_score
            scored_sentences.append((sentence, total_score))
        
        # Sort sentences by score
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Select top sentences until we reach max_length
        selected_sentences = []
        word_count = 0
        
        for sentence, _ in sorted_sentences:
            words_in_sentence = len(sentence.split())
            if word_count + words_in_sentence <= max_length:
                selected_sentences.append(sentence)
                word_count += words_in_sentence
            else:
                break
        
        # Sort selected sentences by original order
        ordered_sentences = []
        for sentence in sentences:
            if sentence in selected_sentences:
                ordered_sentences.append(sentence)
        
        # Create summary
        summary = " ".join(ordered_sentences)
        
        if not summary:
            # Fallback - just return the beginning of the text
            words = text.split()
            summary = " ".join(words[:max_length])
        
        return summary
    
    @staticmethod
    def write_text_to_file(text: str, output_path: str) -> str:
        """
        Write text to a file.
        
        Args:
            text: Text to write
            output_path: Path to the output file
            
        Returns:
            Path to the output file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Text written to {output_path}")
        return output_path
