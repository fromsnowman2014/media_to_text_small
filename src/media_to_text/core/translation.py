"""
Translation functionality using the Hugging Face Transformers library.
"""

import gc
import os
import re
import time
import traceback
import logging
from typing import List, Dict, Optional, Tuple, Any, Union

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

from ..utils.logging_utils import setup_logger
from ..utils.text_splitter import split_text_into_chunks

# Set up logger
logger = setup_logger('media_to_text.translation')

from transformers import M2M100ForConditionalGeneration
# Import only what we need

# Default models for translation
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"
MULTILINGUAL_MODEL = "facebook/m2m100_418M"
SMALL_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M" # Smaller translation model as fallback
TINY_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE" # Tiny model for English to Romance languages
# Use this smaller model which is better suited for CPU deployment
MEMORY_EFFICIENT_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Smaller memory footprint for CPU

# Common language codes mapping (ISO language code to model-specific code)
LANGUAGE_CODE_MAP = {
    'en': 'en',  # English
    'ko': 'ko',  # Korean
    'fr': 'fr',  # French
    'es': 'es',  # Spanish
    'de': 'de',  # German
    'ja': 'jap',  # Japanese (opus-mt uses 'jap')
    'zh': 'zh',  # Chinese
    'ru': 'ru',  # Russian
    'it': 'it',  # Italian
    'pt': 'pt',  # Portuguese
}

class Translator:
    """
    Handles text translation using Hugging Face Transformers models.
    """
    
    def __init__(
        self,
        source_lang: str = 'en',
        target_lang: str = 'ko',
        model_name: Optional[str] = None,
        device: str = 'cpu',
        debug: bool = False
    ):
        """
        Initialize the translator with specified languages and model.
        
        Args:
            source_lang: Source language code (default: 'en')
            target_lang: Target language code (default: 'ko')
            model_name: Model name/path (if None, use default model for the language pair)
            device: Device to use for inference ('cpu' or 'cuda')
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = setup_logger("media_to_text.translation", level='DEBUG')
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_pipeline = None
        self.model = None
        self.tokenizer = None
        
        # Map language codes if needed
        src_code = LANGUAGE_CODE_MAP.get(source_lang, source_lang)
        tgt_code = LANGUAGE_CODE_MAP.get(target_lang, target_lang)
        
        # Select model
        if model_name is None:
            # Use multilingual model for wider language support
            self.model_name = MULTILINGUAL_MODEL
            self.is_multilingual = True
        else:
            # Use specific model
            self.model_name = model_name
            self.is_multilingual = "m2m100" in model_name or "mbart" in model_name or "nllb" in model_name
        
        logger.info(f"Initializing translation model: {self.model_name}")
        logger.info(f"Source language: {source_lang}, Target language: {target_lang}")
        
        try:
            # Configure device
            self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
            if self.device != device and device == 'cuda':
                logger.warning("CUDA not available, falling back to CPU")
            
            # Force garbage collection before loading model
            logger.debug("Running garbage collection before model loading")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Set memory management options to prevent segmentation fault
            logger.debug("Setting torch memory management options")
            torch.set_num_threads(1)  # Limit CPU threads to prevent memory issues
            
            self._load_model_with_progressive_fallback()
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load translation model: {str(e)}")
    
    def _load_model_with_progressive_fallback(self):
        """Attempt to load the translation model using progressive fallback strategies."""
        
        # Strategy 1: Use a plain pipeline with minimal options to maximize compatibility
        try:
            logger.info(f"Strategy 1: Loading translation pipeline with basic config: {self.model_name}")
            self.translation_pipeline = pipeline(
                task="translation",
                model=self.model_name,
                device=self.device,
                # No extras to maximize compatibility
            )
            logger.info("Translation pipeline loaded successfully using basic pipeline approach")
            return
        except Exception as pipeline_err:
            logger.warning(f"Strategy 1 (basic pipeline) failed: {str(pipeline_err)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
        
        # Strategy 2: Load with minimal options, explicit float32 dtype to avoid meta tensors
        try:
            logger.info(f"Strategy 2: Loading with explicit float32 dtype")
            
            # First load tokenizer separately - less prone to errors
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    # Try without safetensors first as many models don't have it
                )
                logger.info(f"Tokenizer loaded successfully")
            except Exception as tok_err:
                logger.warning(f"Failed to load tokenizer: {str(tok_err)}")
                raise
                
            # Set correct source and target language for tokenizer
            if self.is_multilingual:
                if hasattr(self.tokenizer, 'src_lang') and hasattr(self.tokenizer, 'tgt_lang'):
                    self.tokenizer.src_lang = self.source_lang
                    self.tokenizer.tgt_lang = self.target_lang
            
            # Load model with explicit dtype and CPU/float32 settings
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Explicit dtype to avoid meta tensors
                device_map="cpu",           # Force CPU device mapping
                low_cpu_mem_usage=False     # Disable low CPU mem usage which can cause issues
            )
            
            # Force model to CPU explicitly before creating pipeline
            self.model = self.model.to("cpu") 
            
            self.translation_pipeline = pipeline(
                task="translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            
            # Set special tokens and language IDs
            if self.is_multilingual:
                self._setup_multilingual_model()
            
            logger.info("Translation pipeline created successfully with explicit CPU config")
            return
        except Exception as explicit_err:
            logger.warning(f"Strategy 2 (explicit float32/CPU) failed: {str(explicit_err)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
        
        # Strategy 3: Try different, smaller Helsinki-NLP model that should work on CPU
        try:
            # Change to a more CPU-friendly model
            logger.info(f"Strategy 3: Trying a smaller CPU-friendly model (Helsinki-NLP)")
            
            # Select a language-specific Helsinki model when possible
            if self.source_lang == 'en' and self.target_lang in ['es', 'fr', 'it', 'pt']:
                model_name = "Helsinki-NLP/opus-mt-en-ROMANCE" 
            elif self.source_lang == 'en' and self.target_lang in ['de', 'nl']:
                model_name = "Helsinki-NLP/opus-mt-en-de"  
            elif self.source_lang == 'en' and self.target_lang == 'ko':
                model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"  # Use the correct en-ko model
            else:
                # Fallback to general Helsinki model (won't work for all language pairs)
                model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
                
            self.model_name = model_name
            self.is_multilingual = False  # Helsinki models are bilingual/language-group specific
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Explicit dtype
            )
            
            # Force model to CPU explicitly
            self.model = self.model.to("cpu")
            
            self.translation_pipeline = pipeline(
                task="translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cpu",
            )
            
            logger.info(f"Translation pipeline created successfully with alternative model {model_name}")
            return
        except Exception as helsinki_err:
            logger.warning(f"Strategy 3 (Helsinki model) failed: {str(helsinki_err)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            
        # Strategy 4: Fallback to MarianMT which is tiny and should work even on limited CPU
        try:
            logger.info("Strategy 4: Final fallback to MarianMT tiny model")
            
            # MarianMT - extremely small model
            if self.source_lang == 'en' and self.target_lang in ['es', 'fr', 'it', 'pt', 'ro']:
                # These are the only language pairs supported by this fallback
                model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
                self.model_name = model_name
                self.is_multilingual = False
                
                # Use simple loading, minimal parameters
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model.to("cpu")
                
                # Create a basic pipeline
                self.translation_pipeline = pipeline(
                    task="translation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device="cpu",
                )
                
                logger.warning(f"Using minimal fallback model {model_name}. Note: Limited language support!")
                logger.warning(f"Original requested language pair was {self.source_lang}-{self.target_lang}")
                return
            else:
                # Try with development fallback for unsupported languages (last resort)
                logger.warning(f"No direct translation model found for {self.source_lang}-{self.target_lang}")
                logger.warning(f"Creating development fallback for testing purposes")
                
                # Create a pseudo-translation pipeline for development/testing
                self._create_development_fallback_pipeline()
                logger.warning("Using development fallback pipeline (for testing only)")
                return
                
        except Exception as final_err:
            logger.error(f"Final model loading error: {str(final_err)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load any translation model: {str(final_err)}")
    def translate(self, text: str, max_length: int = 512, chunk_size: int = 1000, overlap: int = 100) -> str:
        """
        Translate text from source language to target language.
        
        For long text, this will split the text into chunks, translate each chunk,
        and join the results. Chunks will overlap to maintain context between translations.
        
        Args:
            text: Text to translate
            max_length: Maximum length for generated translation
            chunk_size: Maximum characters per chunk for long text
            overlap: Characters to overlap between chunks for context preservation
            
        Returns:
            Translated text
        """
        # Check for empty input
        if text.strip() == "":
            return ""
        
        # For short text, translate directly
        if len(text) <= chunk_size:
            try:
                return self._translate_text_safely(text, max_length)
            except Exception as e:
                logger.error(f"Translation error for short text: {str(e)}")
                logger.error(traceback.format_exc())
                return text  # Return original text on error
        
        # For longer text, split into chunks and translate each
        logger.info(f"Text length ({len(text)}) exceeds chunk size ({chunk_size}). Splitting into chunks.")
        
        try:
            # Split text into chunks with larger overlap for context preservation
            chunks = split_text_into_chunks(
                text, 
                max_chunk_size=chunk_size, 
                overlap=overlap,  # Increased overlap for better context
                respect_paragraphs=True, 
                respect_sentences=True
            )
            
            logger.info(f"Text split into {len(chunks)} chunks for translation.")
            
            # Preprocess chunks to enhance context preservation
            context_enhanced_chunks = self._enhance_chunks_with_context(chunks)
            logger.debug(f"Enhanced {len(chunks)} chunks with additional context")
            
            # Translate each chunk and join the results
            raw_translated_chunks = []
            
            # For small number of chunks (<= 2), use the existing pipeline
            if len(context_enhanced_chunks) <= 2:
                for i, chunk in enumerate(context_enhanced_chunks):
                    logger.debug(f"Translating chunk {i+1}/{len(context_enhanced_chunks)} ({len(chunk)} characters)")
                    logger.debug(f"Chunk content preview: {chunk[:50]}...")
                    
                    # Force garbage collection between chunks
                    if i > 0:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Translate the current chunk
                    translated_chunk = self._translate_text_safely(chunk, max_length)
                    logger.debug(f"Translated chunk preview: {translated_chunk[:50]}...")
                    raw_translated_chunks.append(translated_chunk)
            else:
                # For larger number of chunks (>2), use multi-chunk strategy with pipeline recreation
                logger.info("Using multi-chunk strategy with pipeline recreation")
                # Save original pipeline/model/tokenizer for restoration
                original_pipeline = self.translation_pipeline
                original_model = self.model
                original_tokenizer = self.tokenizer
                
                try:
                    # Translate each chunk with a fresh pipeline for consistency
                    for i, chunk in enumerate(context_enhanced_chunks):
                        logger.debug(f"Translating chunk {i+1}/{len(context_enhanced_chunks)} ({len(chunk)} chars)")
                        logger.debug(f"Chunk preview: {chunk[:50]}...")
                        
                        # For chunks after the first one, recreate the pipeline to avoid state corruption
                        if i > 0:
                            logger.debug("Creating fresh translation pipeline for chunk consistency")
                            self._create_translation_pipeline()
                        
                        # Translate with the fresh pipeline
                        translated_chunk = self._translate_text_safely(chunk, max_length)
                        logger.debug(f"Translated chunk preview: {translated_chunk[:50]}...")
                        raw_translated_chunks.append(translated_chunk)
                    
                    # Delay to ensure clean pipeline state between chunks
                    time.sleep(0.5)  # Small delay to stabilize
                    
                finally:
                    # Restore original pipeline/model/tokenizer
                    logger.debug("Restoring original translation pipeline")
                    self.translation_pipeline = original_pipeline
                    self.model = original_model
                    self.tokenizer = original_tokenizer
            
            # Post-process translated chunks to remove overlapping content
            processed_chunks = self._post_process_translated_chunks(raw_translated_chunks, chunks)
            
            # Join all processed chunks and return
            result = " ".join(processed_chunks)
            logger.info(f"Translation complete: {len(text)} chars in, {len(result)} chars out")
            return result
            
        except Exception as e:
            logger.error(f"Chunk translation error: {str(e)}")
            logger.error(traceback.format_exc())
            return text  # Return original text on error

    def _create_development_fallback_pipeline(self):
        """
        This is only for testing purposes when no suitable model is available.
        It simulates translation by adding a prefix to the text.
        """
        logger.warning("Creating development-only fallback translation pipeline")
        
        # Create a basic pseudo-translator function for development/testing
        def pseudo_translate_fn(texts, **kwargs):
            """Accept any keyword arguments for compatibility with HF pipeline interface"""
            if isinstance(texts, str):
                texts = [texts]
                
            results = []
            for text in texts:
                # This is just a placeholder that adds language markers for testing
                translated = f"[{self.target_lang}] {text} [{self.source_lang}→{self.target_lang}]"
                results.append({"translation_text": translated})
                
            return results
            
        # Create a simple callable pipeline for compatibility
        self.translation_pipeline = pseudo_translate_fn
        logger.warning("Development fallback translation pipeline created (does not perform real translation)")
    
    def _create_translation_pipeline(self):
        """Recreate the translation pipeline using existing model and tokenizer.
        Used when multiple chunks need fresh pipeline instances.
        """
        logger.debug("Creating fresh translation pipeline instance")
        
        try:
            # If we already have model and tokenizer, use them
            if self.model is not None and self.tokenizer is not None:
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Reset source/target languages if needed
                if self.is_multilingual and hasattr(self.tokenizer, 'src_lang') and hasattr(self.tokenizer, 'tgt_lang'):
                    self.tokenizer.src_lang = self.source_lang
                    self.tokenizer.tgt_lang = self.target_lang
                
                # Create a new pipeline instance with existing model/tokenizer
                self.translation_pipeline = pipeline(
                    task="translation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )
                
                logger.debug("Translation pipeline recreated successfully")
            else:
                # If model/tokenizer not available, use full loading strategy
                logger.debug("Model or tokenizer not available, using full loading strategy")
                self._load_model_with_progressive_fallback()
        except Exception as e:
            logger.error(f"Error recreating translation pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            # If recreation fails, don't reset the pipeline - keep using the existing one
            # This is better than having no pipeline at all
    
    def _enhance_chunks_with_context(self, chunks):
        """
        Enhance chunks with additional context from adjacent chunks to improve translation continuity.
        This helps maintain context between chunk boundaries for better translation quality.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of enhanced chunks with context from adjacent chunks
        """
        if len(chunks) <= 1:
            return chunks
            
        enhanced_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                # First chunk - no previous context to add
                enhanced_chunks.append(chunks[i])
            else:
                # Extract context from previous chunk
                prev_chunk = chunks[i-1]
                current_chunk = chunks[i]
                
                # Get the last sentence from previous chunk as context
                sentences = re.split(r'(?<=[.!?]\s)', prev_chunk)
                if len(sentences) >= 2:
                    context = sentences[-2] + sentences[-1]  # Last two sentences for better context
                else:
                    context = prev_chunk[-200:] if len(prev_chunk) > 200 else prev_chunk  # At least 200 chars
                
                # Add context at beginning of current chunk
                enhanced_chunks.append(f"{context} {current_chunk}")
                
        return enhanced_chunks
        
    def _post_process_translated_chunks(self, translated_chunks, original_chunks):
        """
        Process translated chunks to remove overlapping content and ensure consistency.
        
        Args:
            translated_chunks: List of raw translated chunks
            original_chunks: List of original text chunks before translation
            
        Returns:
            List of processed translated chunks ready for joining
        """
        if len(translated_chunks) <= 1:
            return translated_chunks
            
        processed_chunks = []
        
        for i in range(len(translated_chunks)):
            if i == 0:
                # First chunk - use as is
                processed_chunks.append(translated_chunks[i])
            else:
                # For subsequent chunks, we need to detect and remove overlapping content
                current_chunk = translated_chunks[i]
                
                # Skip first part which likely contains the overlapping content
                # Check the translated chunk length to avoid removing too much
                skip_ratio = 0.2  # Skip ~20% of the beginning of each chunk after the first
                skip_chars = min(int(len(current_chunk) * skip_ratio), 100)
                
                # Ensure we don't skip past sentence boundaries
                # Find the first sentence boundary after skip_chars
                match = re.search(r'[.!?]\s', current_chunk[skip_chars:skip_chars+100])
                if match:
                    skip_chars = skip_chars + match.end()  # Include the punctuation and space
                
                # Add the processed chunk (without overlapping content)
                processed_chunks.append(current_chunk[skip_chars:])
                
        return processed_chunks
    
    def _translate_text_safely(self, text: str, max_length: int = 512) -> str:
        """Helper method to translate text with appropriate handling for each model type."""
        try:
            # Check if translation pipeline is available
            if self.translation_pipeline is None:
                logger.warning("Translation pipeline not available")
                return text
                
            # Use the pipeline approach first (most efficient)
            try:
                logger.debug("Using translation pipeline for translation")
                result = self.translation_pipeline(text, max_length=max_length)
                
                # Handle different return formats from different pipeline implementations
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and 'translation_text' in result[0]:
                        return result[0]['translation_text']
                    else:
                        return result[0]
                elif isinstance(result, dict) and 'translation_text' in result:
                    return result['translation_text']
                else:
                    return str(result)
                    
            except RuntimeError as pipeline_err:
                logger.warning(f"Pipeline translation error: {str(pipeline_err)}. Trying direct model approach.")
            
            # Direct model approach as fallback
            if self.is_multilingual and self.model is not None and self.tokenizer is not None:
                # Handle multilingual model
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'architectures') and \
                   self.model.config.architectures[0] == 'M2M100ForConditionalGeneration':
                    # Process for M2M100
                    logger.debug("Using M2M100 direct model approach")
                    
                    # Don't move model - use inputs compatible with model's current device
                    self.tokenizer.src_lang = self.source_lang
                    self.tokenizer.tgt_lang = self.target_lang
                    
                    # Create tokenized inputs
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    # Get model device from any parameter
                    model_device = next(self.model.parameters()).device
                    logger.debug(f"Model is on device: {model_device}")
                    
                    # Move inputs to model's device
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    try:
                        # Safety wrapper for generate to avoid device mismatches
                        with torch.no_grad():
                            translated_tokens = self.model.generate(
                                **inputs,
                                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                                max_length=max_length
                            )
                        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    except Exception as e:
                        logger.error(f"Error during direct model translation: {str(e)}")
                        # If direct approach failed, try something simpler
                        if not text or not text.strip():
                            return text
                        return f"[Translation Error: {str(e)[:50]}...]\n{text}"
                else:
                    # Process for MBart
                    logger.debug("Using MBart direct model approach")
                    self.tokenizer.src_lang = self.source_lang
                    
                    # Create tokenized inputs
                    inputs = self.tokenizer(text, return_tensors="pt")
                    
                    # Get model device from any parameter
                    model_device = next(self.model.parameters()).device
                    logger.debug(f"Model is on device: {model_device}")
                    
                    # Move inputs to model's device
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    
                    try:
                        # Safety wrapper for generate
                        with torch.no_grad():
                            translated_tokens = self.model.generate(
                                **inputs,
                                forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
                                max_length=max_length
                            )
                        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    except Exception as e:
                        logger.error(f"Error during direct model translation: {str(e)}")
                        # If direct approach failed, try something simpler
                        if not text or not text.strip():
                            return text
                        return f"[Translation Error: {str(e)[:50]}...]\n{text}"
            else:
                # No available translation method
                logger.warning("No translation method available")
                return text
            
            logger.debug(f"Translation result: {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation error in _translate_text_safely: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return text  # Return original text on error
    
    def translate_batch(
        self, 
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            batch_size: Number of texts to translate at once
            max_length: Maximum length of generated translations
            
        Returns:
            List of translated texts
        """
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Translating batch {i//batch_size + 1} ({len(batch)} texts)")
            
            try:
                batch_results = []
                for text in batch:
                    batch_results.append(self.translate(text, max_length=max_length))
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch translation error: {str(e)}")
                # Add original texts on error
                results.extend(batch)
        
        return results
    
    def translate_segments(
        self, 
        segments: List[Dict[str, Any]],
        text_key: str = 'text'
    ) -> List[Dict[str, Any]]:
        """
        Translate text in a list of segment dictionaries.
        
        Args:
            segments: List of segment dictionaries
            text_key: Key for the text field in the segments
            
        Returns:
            List of segments with translated text
        """
        texts = [segment[text_key] for segment in segments if text_key in segment]
        translations = self.translate_batch(texts)
        
        translated_segments = []
        translation_idx = 0
        
        for segment in segments:
            segment_copy = segment.copy()
            if text_key in segment_copy:
                segment_copy[f'translated_{text_key}'] = translations[translation_idx]
                translation_idx += 1
            translated_segments.append(segment_copy)
        
        return translated_segments
    
    def translate_file(
        self, 
        input_path: str,
        output_path: str
    ) -> str:
        """
        Translate text from a file and write the result to another file.
        
        Args:
            input_path: Path to the input text file
            output_path: Path to write the translated text
            
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
        
        # Translate text
        translation = self.translate(text)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write translation to output file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translation)
            logger.info(f"Translation written to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error writing output file: {str(e)}")
            raise RuntimeError(f"Error writing output file: {str(e)}")
