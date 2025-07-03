"""
Text summarization functionality using the Hugging Face Transformers library.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Any, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.summarization")

# Default models for summarization
DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
SMALL_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-6-6" # Smaller model as fallback
MULTILINGUAL_SUMMARIZATION_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

class Summarizer:
    """
    Handles text summarization using Hugging Face Transformers models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_SUMMARIZATION_MODEL,
        language: str = 'en',
        device: str = 'cpu',
        debug: bool = False
    ):
        """
        Initialize the summarizer with specified model and settings.
        
        Args:
            model_name: Model name/path (if None, use default model)
            language: Language code for multilingual models
            device: Device to use for inference ('cpu' or 'cuda')
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = get_logger("media_to_text.summarization", debug_mode=True)
        
        self.language = language
        self.model_name = model_name if model_name else DEFAULT_SUMMARIZATION_MODEL
        self.is_multilingual = "mbart" in self.model_name
        
        logger.info(f"Initializing summarization model: {self.model_name}")
        
        try:
            # Configure device
            self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
            if self.device != device and device == 'cuda':
                logger.warning("CUDA not available, falling back to CPU")
            
            # Try to use pipeline directly with model_name first (safer loading)
            try:
                logger.info(f"Attempting to load summarization model using pipeline: {self.model_name}")
                self.summarizer = pipeline(
                    "summarization", 
                    model=self.model_name,
                    device=0 if self.device == 'cuda' else -1
                )
                self.model = None  # No need to keep separate model reference
                self.tokenizer = None
                
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {str(e)}, trying smaller model...")
                # Try a smaller model as fallback
                try:
                    logger.info(f"Attempting to load smaller summarization model: {SMALL_SUMMARIZATION_MODEL}")
                    self.summarizer = pipeline(
                        "summarization", 
                        model=SMALL_SUMMARIZATION_MODEL,
                        device=0 if self.device == 'cuda' else -1
                    )
                    self.model = None
                    self.tokenizer = None
                except Exception as e2:
                    logger.error(f"Failed to load smaller model: {str(e2)}")
                    raise
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model: {str(e)}")
            raise RuntimeError(f"Failed to load summarization model: {str(e)}")
    
    def _chunk_text(
        self, 
        text: str, 
        max_chunk_size: int = 1024
    ) -> List[str]:
        """
        Split text into chunks for processing long texts.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for the space
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def summarize(
        self, 
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False
    ) -> str:
        """
        Summarize a text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            do_sample: Whether to use sampling for generation
            
        Returns:
            Summarized text
        """
        try:
            logger.debug(f"Summarizing text (length: {len(text)})")
            
            # Handle multilingual models
            if self.is_multilingual and hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = self.language
            
            # Handle long texts by chunking
            if len(text) > 1024:
                chunks = self._chunk_text(text)
                logger.debug(f"Text split into {len(chunks)} chunks")
                
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}")
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length//2,  # Shorter summary for each chunk
                        min_length=min_length//2,
                        do_sample=do_sample
                    )
                    chunk_summaries.append(summary[0]['summary_text'])
                
                # Combine chunk summaries and summarize again
                combined_summary = " ".join(chunk_summaries)
                if len(combined_summary) > 1024:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample
                    )
                    return final_summary[0]['summary_text']
                else:
                    return combined_summary
            else:
                # Summarize directly for shorter texts
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample
                )
                return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            # Return a truncated version of the text as fallback
            return text[:max_length] + "..."
    
    def summarize_segments(
        self, 
        segments: List[Dict[str, Any]],
        text_key: str = 'text',
        max_segments_per_batch: int = 10,
        max_length: int = 150
    ) -> Dict[str, Any]:
        """
        Summarize transcribed segments.
        
        Args:
            segments: List of segment dictionaries
            text_key: Key for the text field in the segments
            max_segments_per_batch: Maximum number of segments to combine for a single summary
            max_length: Maximum length of the summary
            
        Returns:
            Dictionary with full text and summary
        """
        # Combine segment texts
        full_text = " ".join([segment[text_key] for segment in segments if text_key in segment])
        
        # Create batch summaries for very long transcripts
        if len(segments) > max_segments_per_batch:
            batch_summaries = []
            for i in range(0, len(segments), max_segments_per_batch):
                batch_segments = segments[i:i+max_segments_per_batch]
                batch_text = " ".join([s[text_key] for s in batch_segments if text_key in s])
                batch_summary = self.summarize(batch_text, max_length=max_length//2)
                batch_summaries.append(batch_summary)
            
            # Create a final summary from batch summaries
            combined_batch_summaries = " ".join(batch_summaries)
            final_summary = self.summarize(combined_batch_summaries, max_length=max_length)
        else:
            # Summarize directly for shorter transcripts
            final_summary = self.summarize(full_text, max_length=max_length)
        
        return {
            "full_text": full_text,
            "summary": final_summary
        }
    
    def summarize_file(
        self, 
        input_path: str,
        output_path: str,
        max_length: int = 150
    ) -> str:
        """
        Summarize text from a file and write the result to another file.
        
        Args:
            input_path: Path to the input text file
            output_path: Path to write the summary
            max_length: Maximum length of the summary
            
        Returns:
            Path to the output file
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Read input file
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            raise RuntimeError(f"Error reading input file: {str(e)}")
        
        # Summarize text
        summary = self.summarize(text, max_length=max_length)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write summary to output file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Summary written to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error writing output file: {str(e)}")
            raise RuntimeError(f"Error writing output file: {str(e)}")
