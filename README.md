# Media-to-Text CLI Tool

A powerful command-line tool for transcribing, translating, summarizing, and subtitle generation from audio/video files. The tool leverages state-of-the-art AI models for high-quality text extraction and processing.

## Features

- **Transcription**: Convert audio/video files to text using Faster Whisper
- **Direct Text Processing**: Directly translate and summarize text from `.txt`, `.srt`, and `.pdf` files.
- **Translation**: Translate transcribed or input text to different languages using Hugging Face Transformers
- **Summarization**: Generate concise summaries of transcripts, with context-aware summarization for different input types (time-based for media, section-based for documents).
- **Subtitle Generation**: Create SRT and VTT subtitle files from transcriptions (for audio/video inputs only).
- **Batch Processing**: Process multiple files in a directory
- **Flexible Output**: Save results to customizable locations

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/fromsnowman2014/media_to_text_summary.git
cd media_to_text_summary
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# For PDF processing, you may need to install system-level dependencies for pdfplumber
# pip install "pdfplumber[image]"
```

## Usage

### Basic Usage

Due to OpenMP conflicts between PyTorch and faster-whisper, you should use one of the following methods to run the application:

#### Method 1: Using the provided wrapper script

```bash
./run.sh input_file.mp3
```

#### Method 2: Setting the environment variable directly

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m extract_transcript.cli input_file.mp3
```

This will transcribe the audio file and save the result to the `output` directory.

### Advanced Options

```bash
# Using the wrapper script
./run.sh input_file.mp3 \
    --model medium \
    --language en \
    --output_dir my_output \
    --translate_to ko \
    --summarize \
    --summary_length 200 \
    --generate_subtitles \
    --subtitle_format both

# Or with the environment variable
KMP_DUPLICATE_LIB_OK=TRUE python -m extract_transcript.cli input_file.mp3 \
    --model medium \
    --language en \
    --output_dir my_output \
    --translate_to ko \
    --summarize \
    --summary_length 200 \
    --generate_subtitles \
    --subtitle_format both
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Path to audio/video/text/PDF file or directory | Required |
| `--input_type` | Explicitly specify input type (`audio`, `video`, `transcript`, `pdf`) | Auto-detect from extension |
| `--model` | Whisper model size (tiny, base, small, medium, large). Not used for text/PDF. | base |
| `--language` | Language code for transcription/text | Auto-detect |
| `--output_dir` | Output directory for results | `output/` |
| `--translate_to` | Target language for translation | None |
| `--summarize` | Generate a summary of the transcript/text | False |
| `--summary_length` | Maximum summary length in words | 150 |
| `--generate_subtitles` | Generate subtitle files (media files only) | False |
| `--subtitle_format` | Subtitle format (srt, vtt, or both) | srt |

## Directory Processing

Process all supported audio/video files in a directory:

```bash
# Using the wrapper script
./run.sh path/to/media/folder --generate_subtitles

# Or with the environment variable
KMP_DUPLICATE_LIB_OK=TRUE python -m extract_transcript.cli path/to/media/folder --generate_subtitles
```

## Architecture

The codebase follows SOLID design principles with separate modules for each responsibility:

- **Transcriber**: Handles audio/video to text conversion
- **Translator**: Manages text translation between languages
- **Summarizer**: Creates concise summaries from transcripts
- **SubtitleGenerator**: Converts segments to SRT/VTT formats
- **OutputWriter**: Manages file system operations for output
- **CLI**: Coordinates the workflow and user interaction

All these modules are located in the `extract_transcript` package.

## Supported Formats

### Input Formats
- Audio: mp3, wav, flac, ogg, m4a
- Video: mp4, mov
- Text: txt, srt
- Document: pdf

### Output Formats
- Transcription: Plain text (.txt)
- Translation: Plain text (.txt)
- Summary: Plain text (.txt)
- Subtitles: SRT (.srt) and/or WebVTT (.vtt)

## Development

### Known Issues

#### OpenMP Conflicts
The application uses both PyTorch and faster-whisper, which can cause OpenMP runtime conflicts. This is resolved by setting the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable as shown in the usage examples.

#### PyTorch Loading Security
To address a security vulnerability in `torch.load`, we use the `safetensors` format for loading models by default. The safetensors package is included in the requirements.txt file.

### Testing

Run the test suite:

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest -xvs
```

### Adding New Features

The modular architecture makes it easy to extend functionality:

1. Add a new module in the `extract_transcript` package
2. Update the CLI interface in `cli.py`
3. Write unit tests in the `tests` directory

## License

[MIT License](LICENSE)
