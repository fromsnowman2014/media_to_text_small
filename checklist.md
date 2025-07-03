# **Development Checklist: Media-to-Text Converter**

This checklist is based on the development plan and is designed to track the progress of each implementation step.

---

## **Part 1: Core Library & CLI Development**

### **Step 1: Project Structure & Initial Setup**
- [ ] Create the main project directory structure (`media_to_text/`, `tests/`, `scripts/`).
- [ ] Create empty core module files: `processor.py`, `transcriber.py`, `translator.py`, `summarizer.py`, `subtitle_generator.py`, `file_handler.py`.
- [ ] Create the empty utility file: `utils/text_splitter.py`.
- [ ] Create the main entry point `cli.py`.
- [ ] Create and populate `requirements.txt` with necessary libraries.
- [ ] Set up a Python virtual environment and install dependencies.
- [ ] Create the `run.sh` script with the `KMP_DUPLICATE_LIB_OK=TRUE` variable.

### **Step 2: Define Workflow Controller and Data Structures**
- [ ] Define the main `process_file` function signature in `core/processor.py`.
- [ ] Define the data structures (e.g., Data Class or TypedDict) for passing data between modules.

### **Step 3: Implement the Transcription Module**
- [ ] Implement the transcription class/function in `core/transcriber.py` using `faster-whisper`.
- [ ] Write a unit test for the transcription module using a sample audio file.
- [ ] Confirm that the system has FFmpeg installed or add instructions in the `README.md`.

### **Step 4: Implement the Translation Module**
- [ ] Implement the `split_text_into_chunks` function in `utils/text_splitter.py`.
- [ ] Implement the core translation logic (chunking, translating, merging) in `core/translator.py`.
- [ ] Write a unit test to verify the translation of a long text document.

### **Step 5: Implement Summarization and Subtitle Generation Modules**
- [ ] Implement the summarization function in `core/summarizer.py`.
- [ ] Write a unit test for the summarization function.
- [ ] Implement the SRT/VTT generation function in `core/subtitle_generator.py`.
- [ ] Write a unit test for the subtitle generation function.

### **Step 6: Implement CLI Interface and Final Assembly**
- [ ] Implement all command-line argument parsing in `cli.py` using `argparse`.
- [ ] Connect the CLI arguments to the `core/processor.py` to trigger the workflow.
- [ ] Integrate `tqdm` for a progress bar in the CLI.
- [ ] Perform a full integration test by running the CLI with various options.

---

## **Part 2: macOS GUI Application Development**

### **Step 7: Design GUI Structure and Asynchronous Processing**
- [ ] Create the `media_to_text/gui/` directory.
- [ ] Design the `Worker` class using `QThread` to handle background processing.
- [ ] Design the basic layout of the `QMainWindow` for the application.

### **Step 8: Implement GUI Widgets and Connect to Core Logic**
- [ ] Implement all required UI widgets (file list, buttons, dropdowns, checkboxes).
- [ ] Connect the 'Start Processing' button to instantiate and run the `Worker` thread.
- [ ] Implement signal/slot connections for updating the UI (progress bar, status messages) from the `Worker` thread.

### **Step 9: Enhance User Experience (UX)**
- [ ] Implement drag-and-drop functionality for the file list.
- [ ] Implement the 'Show in Finder' feature.
- [ ] Add user-friendly dialog boxes for error reporting.

---

## **Part 3: Finalization & Deployment**

### **Step 10: Code Refactoring and Documentation**
- [ ] Review and refactor the entire codebase for clarity and performance.
- [ ] Add comprehensive Docstrings and type hints to all functions and classes.
- [ ] Update the `README.md` with detailed instructions for both CLI and GUI versions.

### **Step 11: Integration Testing and Release**
- [ ] Perform final end-to-end testing with a wide range of real-world media files.
- [ ] Create a final version tag in `git` (e.g., `v1.0`).