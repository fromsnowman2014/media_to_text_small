"""
Translation functionality using the Hugging Face Transformers library.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.translation")

# Default models for translation
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"
MULTILINGUAL_MODEL = "facebook/m2m100_418M"
SMALL_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M" # Smaller translation model as fallback

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
            
            # Try to load translation model safely using pipeline directly
            try:
                logger.info(f"Attempting to load translation model using pipeline: {self.model_name}")
                self.translator = pipeline(
                    "translation", 
                    model=self.model_name,
                    device=0 if self.device == 'cuda' else -1
                )
                self.model = None  # No need to keep separate model reference
                self.tokenizer = None
                
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {str(e)}, trying smaller model...")
                # Try with smaller model as fallback
                try:
                    logger.info(f"Attempting to load smaller translation model: {SMALL_TRANSLATION_MODEL}")
                    self.translator = pipeline(
                        "translation", 
                        model=SMALL_TRANSLATION_MODEL,
                        src_lang=self.source_lang,
                        tgt_lang=self.target_lang,
                        device=0 if self.device == 'cuda' else -1
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
        try:
            logger.debug(f"Translating text: {text[:50]}...")
            
            if self.is_multilingual:
                # For multilingual models
                if hasattr(self.tokenizer, 'src_lang') and hasattr(self.tokenizer, 'tgt_lang'):
                    # For MBart
                    self.tokenizer.src_lang = self.source_lang
                    self.tokenizer.tgt_lang = self.target_lang
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    translated_tokens = self.model.generate(
                        **inputs, 
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                        max_length=max_length
                    )
                    translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                else:
                    # For M2M100
                    self.tokenizer.src_lang = self.source_lang
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    translated_tokens = self.model.generate(
                        **inputs, 
                        forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
                        max_length=max_length
                    )
                    translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            else:
                # For language-pair specific models
                translation = self.translator(text, max_length=max_length)[0]['translation_text']
            
            logger.debug(f"Translation result: {translation[:50]}...")
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
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
