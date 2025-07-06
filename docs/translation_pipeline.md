# Translation Pipeline Documentation

## Overview

The translation pipeline in `media_to_text` provides robust text translation capabilities with fallback strategies to handle various environments and resource constraints. It includes text chunking for long content and multiple model loading strategies for stability.

## Features

- **Progressive Model Loading**: Multiple strategies for loading translation models based on available resources
- **Text Chunking**: Automatic splitting of long texts while preserving semantic boundaries
- **Development Fallback**: Pseudo-translation for development/testing when no suitable model is available
- **Batch Translation**: Process texts in batches to manage memory usage
- **Error Handling**: Detailed logging and fallback to simpler models when needed

## Loading Strategies

The translation pipeline implements a progressive fallback approach for model loading:

1. **Basic Pipeline**: First attempts to load the model using the standard HF pipeline API
2. **Explicit CPU Configuration**: Loads with explicit float32 dtype and CPU device mapping
3. **Alternative Models**: Falls back to language-specific Helsinki-NLP models when available
4. **Development Fallback**: For unsupported language pairs, uses a pseudo-translator for testing

## Usage

```python
from media_to_text.core.translation import Translator

# Basic usage
translator = Translator(source_lang='en', target_lang='es')
translated_text = translator.translate("Hello world")

# With chunking for long texts
long_text = open('long_document.txt', 'r').read()
translated = translator.translate(
    long_text, 
    chunk_size=1000,  # Characters per chunk
    overlap=100       # Overlap between chunks
)

# Batch translation
texts = ["Text 1", "Text 2", "Text 3"]
translations = translator.translate_batch(texts, batch_size=8)
```

## Supported Languages

- Primary support through multilingual models: `facebook/m2m100_418M`
- Fallback through smaller models: `Helsinki-NLP/opus-mt-*` family
- For unsupported language pairs, a development fallback provides pseudo-translation

## Known Limitations

1. **Memory Requirements**: Large models like `m2m100_418M` require substantial memory
2. **Device Compatibility**: Meta tensor issues may occur with certain PyTorch versions
3. **Translation Quality**: Fallback models may provide lower quality translations
4. **Language Support**: The Helsinki-NLP fallback has limited language pair support

## Production Recommendations

1. **Environment**: Set memory management options to avoid segmentation faults:
   ```
   OMP_NUM_THREADS=1
   TOKENIZERS_PARALLELISM=false
   ```

2. **Pre-download Models**: For offline use, download models in advance using the HF cache

3. **Resource Allocation**: For large models, ensure adequate RAM (8GB+ recommended)

4. **Chunking Parameters**: Adjust chunk_size and overlap based on content type:
   - Narrative text: larger chunks (1000-2000 chars), smaller overlap (50-100 chars)
   - Technical/code text: smaller chunks (500-1000 chars), larger overlap (100-150 chars)

5. **Error Handling**: Always implement application-level fallbacks for translation errors

## Development Testing

For development or testing purposes, unsupported language pairs will use a pseudo-translation that prefixes text with target language code. This allows testing of the chunking and pipeline logic without requiring all language models.
