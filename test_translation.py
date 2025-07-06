#!/usr/bin/env python3
import sys
import logging
import os
from src.media_to_text.core.translation import Translator
from src.media_to_text.utils.logging_utils import get_logger

# Configure logging
logger = get_logger("test_translation", debug_mode=True)

def main():
    # Set environment variables to help with memory management
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    logger.info("Testing translation with progressive model loading and text chunking")
    
    # Create a translator instance (English to Korean)
    logger.info("Creating translator instance...")
    translator = Translator(source_lang='en', target_lang='ko', debug=True)
    logger.info("Translator created successfully")
    
    # Test with short text (no chunking needed)
    short_text = "Hello, this is a short text for testing translation."
    logger.info(f"Translating short text: '{short_text}'")
    translated_short = translator.translate(short_text)
    logger.info(f"Translated short text: '{translated_short}'")
    
    # Test with longer text that triggers chunking
    long_text = """
    This is a longer text that should trigger the text chunking functionality.
    
    It has multiple paragraphs to test that the chunking preserves paragraph boundaries.
    Each paragraph contains multiple sentences. The sentences should be preserved during chunking.
    This ensures that the semantic meaning is maintained throughout the translation process.
    
    The progressive model loading strategy should handle this text efficiently.
    If the primary model fails to load, it should fall back to alternative strategies.
    This test verifies that our implementation can handle longer texts properly.
    
    Chunking parameters can be adjusted if needed, but the default values should work well.
    The overlap between chunks ensures context is preserved during translation.
    """
    
    logger.info(f"Translating longer text ({len(long_text)} characters)...")
    translated_long = translator.translate(long_text, chunk_size=100, overlap=20)  # Small chunk size for testing
    logger.info(f"Translation of longer text completed")
    logger.info(f"Translated long text:\n{translated_long}")
    
    logger.info("Translation test completed successfully")

if __name__ == "__main__":
    main()
