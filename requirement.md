# **Requirements Specification: Media-to-Text Converter (CLI & macOS GUI)**

## 1. Overview

This document specifies the requirements for a comprehensive media-to-text application designed to run entirely offline on macOS. The application will transcribe audio and video, translate the resulting text, generate summaries, and create subtitles. The primary goal is to provide a powerful tool that is performant even on older hardware with limited resources, without needing an internet connection.

The project will be developed in two main phases:
1.  **Phase 1: Core Command-Line Interface (CLI):** A robust backend that handles all processing logic.
2.  **Phase 2: macOS GUI Application:** An intuitive graphical interface that wraps the CLI's functionality for ease of use.

---

## 2. Core Functionality (CLI & GUI Common)

### 2.1. High-Performance Transcription
* **Engine:** Utilize the `faster-whisper` library to perform fast and accurate speech-to-text conversion locally.
* **Offline Capability:** All transcription models will be stored and run on the local machine.
* **Language Detection:** The system will be capable of automatically detecting the language of the source audio or allow the user to specify it manually.

### 2.2. Robust Multilingual Translation
* [cite_start]**Problem:** Initial models (`NLLB-200-distilled-600M`) have shown performance issues and model loading errors on older hardware, especially with long-form text[cite: 1].
* **Core Requirement:** The system must perform reliable, offline text translation, specifically optimized for environments with limited computational resources.
* **Text Processing Strategy:**
    * [cite_start]**Text Chunking:** To ensure stability and prevent model overload, the system must automatically split long transcripts or documents into smaller, coherent chunks (e.g., paragraphs or sentence groups) before translation[cite: 1]. [cite_start]This avoids exceeding the input token limits of lightweight models[cite: 1].
    * [cite_start]**Iterative Translation:** Each chunk shall be translated sequentially[cite: 1].
    * [cite_start]**Content Merging:** The translated chunks must be intelligently merged back into a single, cohesive text document, preserving the original structure as much as possible[cite: 1].
* [cite_start]**Model Selection:** The system should use a lightweight, open-source translation model (e.g., smaller `NLLB` variants) that is proven to run efficiently on CPU[cite: 1].

### 2.3. Context-Aware Summarization
* **Functionality:** Generate a concise summary of the input text.
* **Input-Specific Logic:**
    * **Media Files (Audio/Video):** The summary should be time-based, outlining key points as they occurred in the recording.
    * **Text/PDF Files:** The summary should be section-based, condensing the main ideas from the document's structure.

### 2.4. Subtitle Generation
* **Functionality:** Create subtitle files from the transcribed text.
* **Supported Formats:** The system must be able to generate both `.srt` and `.vtt` file formats.
* **Applicability:** This feature applies only to audio and video inputs where timestamp data is available.

### 2.5. Diverse File Format Support
* **Input Formats:**
    * **Audio:** `mp3`, `wav`, `m4a`, `flac`, `ogg`
    * **Video:** `mp4`, `mov`, `avi` (audio will be automatically extracted using FFmpeg)
    * **Text:** `.txt`, `.srt` (for direct processing)
    * **Document:** `pdf` (for text extraction and processing)
* **Output Formats:**
    * **Transcription:** `_transcription.txt`
    * **Translation:** `_translated_<lang>.txt`
    * **Summary:** `_summary.txt`
    * **Subtitles:** `.srt`, `.vtt`

---

## 3. Phase 1: Core Command-Line Interface (CLI) Specification

### 3.1. CLI Arguments

| Argument | Description | Default | Status |
|---|---|---|---|
| `input_file` | Path to the source audio/video/text/PDF file or a directory for batch processing. | **Required** | Implemented |
| `--input_type` | Explicitly set input type (`audio`, `video`, `transcript`, `pdf`). | Auto-detect | Implemented |
| `--model` | Whisper model size (e.g., `tiny`, `base`, `small`, `medium`, `large`). | `base` | Implemented |
| `--language` | Language code for transcription (e.g., `en`, `ko`). | Auto-detect | Implemented |
| `--output_dir` | Directory to save the output files. | `output/` | Implemented |
| `--translate_to` | Target language code for translation. If not provided, no translation is performed. | `None` | Implemented |
| `--summarize` | Flag to generate a summary of the content. | `False` | Implemented |
| `--summary_length` | The maximum length of the summary in words. | `150` | Implemented |
| `--generate_subtitles`| Flag to generate subtitle files (for media inputs only). | `False` | Implemented |
| `--subtitle_format` | Subtitle format (`srt`, `vtt`, or `both`). | `srt` | Implemented |

### 3.2. CLI Features
* **Batch Processing:** If the `input_file` is a directory, the tool will process all supported files within it.
* **Status Visualization:** Use `tqdm` to display a progress bar during file processing.
* **Environment Handling:** Address OpenMP conflicts between PyTorch and faster-whisper by using the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable.

---

## 4. Phase 2: macOS GUI Application Specification

### 4.1. Development Goal & Technology
* **Goal:** Provide a user-friendly graphical interface for all core functionalities, enabling non-technical users to leverage the tool's power.
* **Technology:** Use **PySide6** to create a modern, responsive UI that integrates well with macOS.

### 4.2. UI/UX Components
* **Main Window:**
    * **File List Management:**
        * An area to display files queued for processing.
        * "Add File(s)" and "Add Folder" buttons.
        * Support for **Drag and Drop** to add files.
        * Buttons to remove selected files or clear the entire list.
    * **Processing Options Panel:**
        * **Model:** Dropdown to select the Whisper model size.
        * **Language:** Dropdown for the source language (including "Auto-Detect").
        * **Tasks (Checkboxes):**
            * `[ ] Translate`: When checked, reveals a dropdown to select the target language.
            * `[ ] Generate Summary`: Enables summarization.
            * `[ ] Generate Subtitles`: Enabled only when media files are in the list.
    * **Output Location:**
        * A text field displaying the selected output directory.
        * A "Change..." button to open a native folder selection dialog.
    * **Execution and Status:**
        * A "Start Processing" button to begin the tasks.
        * A progress bar showing the overall progress of the batch.
        * A status label indicating the current file and operation (e.g., "Transcribing `interview.mp4`...").
        * A "Show in Finder" link that appears upon completion.

### 4.3. Key UX Considerations
* **Asynchronous Processing:** All file operations must run in a background thread (**QThread**) to keep the UI responsive and prevent freezing.
* **Clear Feedback:** The user must be kept informed about the application's state (idle, processing, success, error) through visual cues.
* **macOS Integration:** The application should feel native by using system file dialogs, respecting settings like Dark Mode, and providing a clean user experience.

---

## 5. Execution Environment & Dependencies

* **Python:** 3.9+
* **System Dependencies:**
    * **FFmpeg:** Must be installed on the system for audio/video processing.
* **Core Python Libraries:**
    * `faster-whisper`
    * `transformers` & `torch`
    * `sentencepiece`
    * `pdfplumber`
    * `safetensors`
* **GUI Python Libraries (Phase 2):**
    * `PySide6`