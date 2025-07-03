# **Development Plan: Media-to-Text Converter**

## 1. Development Principles and Goals

This document defines the concrete execution plan for developing the CLI and macOS GUI application based on the requirements specification. The development will adhere to the following principles:

* **Top-Down Design:** The application's overall architecture and the interfaces between modules will be defined first, before implementing the detailed functionalities. This approach minimizes structural changes during development.
* **Modular Design:** Responsibilities will be separated into distinct files to follow the single-responsibility principle.
    * **Guideline:** To enhance maintainability, functions should be around 50 lines, a class should have one clear responsibility, and a file should contain only a few highly related classes or functions.
* **Incremental Development:** Each step will focus on implementing a single core feature. We will proceed to the next step only after ensuring the current feature is stable.
* **Test-Driven Development (TDD):** For each implemented module, corresponding test code will be written to verify its functionality and ensure stability.

---

## **Part 1: Core Library & CLI Development**

First, we will develop a shared core library that contains the application's business logic, and then build a CLI that utilizes this library.

### **Step 1: Project Structure Design & Initial Setup**

We will begin by designing the project's skeleton to maintain a consistent structure throughout the development process.

* **Action:**
    1.  **Create Directory Structure:**
        ```
        media_to_text_converter/
        ├── media_to_text/    # Core logic package
        │   ├── core/         # Core modules for each feature
        │   ├── utils/        # Helper utilities
        │   ├── cli.py        # CLI interface
        │   └── main.py       # Program entry point
        ├── tests/              # Test code
        └── scripts/            # Useful scripts (e.g., run.sh)
        ```
    2.  **Create Core Module Files (leave them empty for now):**
        * `core/file_handler.py`: Handles file reading (pdf, txt, etc.) and type detection.
        * `core/transcriber.py`: Handles speech-to-text conversion.
        * `core/translator.py`: Handles text translation.
        * `core/summarizer.py`: Handles text summarization.
        * `core/subtitle_generator.py`: Handles subtitle file creation.
        * `core/processor.py`: A workflow controller that orchestrates all processing steps.
        * `utils/text_splitter.py`: A utility for splitting long text for translation.
    3.  **Dependency Management:**
        * Create a `requirements.txt` file and list core libraries such as `faster-whisper`, `transformers`, `torch`, and `pdfplumber`.
        * Set up a Python virtual environment and install the dependencies.
    4.  **Execution Script:**
        * [cite_start]Create a `run.sh` shell script that includes the `KMP_DUPLICATE_LIB_OK=TRUE` environment variable[cite: 1].

### **Step 2: Define Workflow Controller and Data Structures**

Define a central controller to manage the data flow between modules and establish a standard data format.

* **Action:**
    1.  In `core/processor.py`, define the function signature for `process_file`. This function will take user requests (e.g., for translation, summarization) as arguments and manage the entire processing flow.
    2.  Define data classes or a `TypedDict` to be passed between processing steps (e.g., `TranscriptionResult(text: str, segments: list)`).

### **Step 3: Implement the Transcription Module**

Implement the first core feature: the speech-to-text module.

* **Action:**
    1.  In `core/transcriber.py`, implement a class/function using `faster-whisper` that takes an audio/video file path and returns the transcribed text and timestamps (`segments`).
    2.  **Required External Program:** This feature requires **FFmpeg** to be installed on the system.
* **Test:** Write a unit test to verify that a sample audio file (`mp3`) produces the expected text output.

### **Step 4: Implement the Translation Module**

Implement a stable translation module that considers hardware constraints.

* **Action:**
    1.  In `utils/text_splitter.py`, implement a `split_text_into_chunks` function that splits long text while preserving semantic consistency by respecting paragraph and sentence boundaries.
    2.  In `core/translator.py`, implement the logic to sequentially translate the text chunks and then merge the results back into a single text.
    3.  Use a lightweight `transformers` translation model for this feature.
* **Test:** Write a unit test to confirm that a long sample text is correctly split, translated, and merged, yielding the correct final output.

### **Step 5: Implement Summarization and Subtitle Generation Modules**

Add the remaining core features: summarization and subtitle generation.

* **Action:**
    1.  In `core/summarizer.py`, implement the functionality to take text as input and return a summary.
    2.  In `core/subtitle_generator.py`, implement the functionality to take timestamp data (`segments`) and generate a string in `.srt` or `.vtt` format.
* **Test:** Write unit tests for each module to verify the accuracy of their functions.

### **Step 6: Implement CLI Interface and Final Assembly**

Integrate all the developed modules to complete the CLI.

* **Action:**
    1.  [cite_start]In `cli.py`, use `argparse` to write the code for parsing all CLI arguments defined in the requirements specification[cite: 1].
    2.  Based on the parsed arguments, call the `process_file` function in `core/processor.py` to execute the entire workflow.
    3.  Integrate `tqdm` to display a progress bar for file processing.
* **Test:** Perform integration testing by running the CLI with various combinations of options on actual files, and verify that the expected output files are created in the correct location with the correct content.

---

## **Part 2: macOS GUI Application Development**

Develop a user-friendly GUI on top of the completed core library.

### **Step 7: Design GUI Structure and Asynchronous Processing**

Create the GUI's skeleton and design an asynchronous architecture to prevent the UI from freezing during long-running tasks.

* **Action:**
    1.  Create a `media_to_text/gui/` directory to house the GUI-related code.
    2.  Using `PySide6` and `QThread`, design a `Worker` class that will run the core processing logic (the `processor` from Part 1) in a background thread.
    3.  Design the basic UI layout for the main window (`QMainWindow`), including the file list, settings panel, and buttons.

### **Step 8: Implement GUI Widgets and Connect to Core Logic**

Implement the user interface and connect it to the background worker.

* **Action:**
    1.  Implement all UI widgets defined in the requirements specification (dropdowns, checkboxes, progress bar, etc.).
    2.  Connect the logic so that when the 'Start Processing' button is clicked, the settings from the UI are passed to the `Worker` thread to begin the task.
    3.  Implement the logic to receive signals from the `Worker` thread (e.g., progress, status messages) and update the UI in real-time.

### **Step 9: Enhance User Experience (UX)**

Add additional features to improve usability.

* **Action:**
    1.  Implement **Drag and Drop** functionality for the file list area.
    2.  Add a 'Show in Finder' link that appears after a task is complete, allowing users to open the output folder directly.
    3.  Implement dialog boxes to display clear messages to the user in case of an error.

---

## **Part 3: Finalization & Deployment**

### **Step 10: Code Refactoring and Documentation**

Improve the project's overall quality.

* **Action:**
    1.  Refactor the entire codebase to improve readability and eliminate redundancy.
    2.  Add Docstrings and type hints to all functions and classes.
    3.  Update the main `README.md` file with detailed usage instructions for both the CLI and GUI for end-users.

### **Step 11: Integration Testing and Release**

* **Action:**
    1.  Test the entire system (CLI and GUI) with a variety of real-world media files of different types, sizes, and languages.
    2.  Use `git tag` to create a final version release (e.g., `v1.0`).