"""
Translation functionality using the Hugging Face Transformers library.
"""

import os
import logging
import traceback
import gc
from typing import List, Dict, Any, Union, Tuple, Optional

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration
)
# Import only what we need

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.translation")

# Default models for translation
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"
MULTILINGUAL_MODEL = "facebook/m2m100_418M"
SMALL_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M" # Smaller translation model as fallback
TINY_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE" # Tiny model for English to Romance languages

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
            logger = get_logger("media_to_text.translation", debug_mode=True)
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        
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
            self.is_multilingual = "m2m100" in model_name or "mbart" in model_name
        
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
            
            # Try to load translation model safely using explicit model loading rather than pipeline
            try:
                logger.debug(f"Starting model load process for: {self.model_name}")
                try:
                    logger.info(f"Loading tokenizer first: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_safetensors=True)
                    logger.debug(f"Tokenizer loaded successfully")
                    
                    # Set correct source and target language for the tokenizer
                    if self.is_multilingual:
                        logger.debug(f"Setting source language: {self.source_lang}, target language: {self.target_lang}")
                        src_code = self.source_lang
                        tgt_code = self.target_lang
                        if hasattr(self.tokenizer, 'src_lang') and hasattr(self.tokenizer, 'tgt_lang'):
                            self.tokenizer.src_lang = src_code
                            self.tokenizer.tgt_lang = tgt_code
                            logger.debug(f"Tokenizer language attributes set")
                
                    # Load model with low memory footprint options
                    logger.info(f"Loading model with low memory settings: {self.model_name}")
                    # Load model with memory-efficient settings (keep low_cpu_mem_usage=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=True,  # Keep True for memory efficiency
                        use_safetensors=True,
                        torch_dtype=torch.float32,
                        device_map='auto'  # Let the model decide best device mapping
                    )
                    
                    # Create pipeline after successful individual component loading
                    logger.debug(f"Creating translation pipeline from loaded components")
                    self.translator = pipeline(
                        "translation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=-1  # Force CPU usage for stability
                    )
                    logger.info(f"Translation model loaded successfully: {self.model_name}")
                except Exception as e:
                    logger.error(f"Error during explicit model loading: {str(e)}")
                    logger.debug(f"Stack trace: {traceback.format_exc()}")
                    raise e
                
            except Exception as e:
                logger.warning(f"Model loading failed: {str(e)}, trying smaller model...")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                
                # Force garbage collection before trying again
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Try with smaller model as fallback, with extra safety measures
                try:
                    logger.info(f"Attempting to load smaller translation model: {SMALL_TRANSLATION_MODEL}")
                    
                    # First try to only load the tokenizer
                    try:
                        logger.debug("Loading tokenizer for smaller model")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            SMALL_TRANSLATION_MODEL, 
                            use_safetensors=True
                        )
                        logger.debug("Smaller model tokenizer loaded successfully")
                    except Exception as tokenizer_err:
                        logger.error(f"Failed to load smaller model tokenizer: {str(tokenizer_err)}")
                        raise tokenizer_err
                        
                    # Then try to load the model with conservative memory settings
                    try:
                        logger.debug("Loading smaller model with conservative memory settings")
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            SMALL_TRANSLATION_MODEL,
                            low_cpu_mem_usage=True,  # Keep memory efficient loading
                            use_safetensors=True,
                            torch_dtype=torch.float32,
                            device_map='auto'  # Let the model decide best device mapping
                        )
                        logger.debug("Smaller model loaded successfully")
                    except Exception as model_err:
                        logger.error(f"Failed to load smaller model: {str(model_err)}")
                        raise model_err
                    
                    # Create pipeline with the loaded components
                    logger.debug("Creating pipeline with loaded smaller model components")
                    self.translator = pipeline(
                        "translation", 
                        model=self.model,
                        tokenizer=self.tokenizer,
                        src_lang=self.source_lang,
                        tgt_lang=self.target_lang,
                        device=0 if self.device == 'cuda' else -1  # Use numeric device indices
                    )
                    self.model = None
                    self.tokenizer = None
                except Exception as e2:
                    logger.error(f"Failed to load smaller model: {str(e2)}")
                    raise
                
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            raise RuntimeError(f"Failed to load translation model: {str(e)}")
    
    def translate(
        self, 
        text: str,
        max_length: int = 512
    ) -> str:
        """
        Translate a single text.
        
        Args:
            text: Text to translate
            max_length: Maximum length of generated translation
            
        Returns:
            Translated text
        """
        if self.translator is None:
            logger.warning("Translator not available, returning original text")
            return text
        
        if not text or not text.strip():
            return text
            
        try:
            logger.info(f"Translating text ({len(text)} chars)")
            
            # Split long texts to prevent memory issues
            if len(text) > 1000:
                logger.debug("Text is long, splitting into chunks for safer translation")
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                translated_chunks = []
                
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Translating chunk {i+1}/{len(chunks)}")
                    translated_chunk = self._translate_text_safely(chunk, max_length)
                    translated_chunks.append(translated_chunk)
                    
                translation = ' '.join(translated_chunks)
                return translation
            else:
                return self._translate_text_safely(text, max_length)
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            logger.error(traceback.format_exc())
            return text  # Return original text on error
    
    def _translate_text_safely(self, text: str, max_length: int = 512) -> str:
        """Helper method to translate text with appropriate handling for each model type."""
        try:
            # Always use the pipeline approach first, which handles device management internally
            if self.translator is not None:
                logger.debug("Using pipeline approach for translation")
                try:
                    translation = self.translator(text, max_length=max_length)[0]['translation_text']
                    return translation
                except RuntimeError as e:
                    logger.warning(f"Pipeline translation error: {str(e)}. Trying direct model approach.")
                    # Fall through to direct model approach
            
            # Direct model approach as fallback
            if self.is_multilingual and self.model is not None:
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
