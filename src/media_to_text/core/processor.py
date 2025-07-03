"""
Central processor for media-to-text conversion.
Coordinates all core modules for transcription, translation, summarization,
and subtitle generation.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple

from tqdm import tqdm
import traceback

from ..utils.logging_utils import get_logger
from ..utils.file_utils import detect_file_type, get_output_path, InputType
from .transcription import Transcriber
from .media_processing import extract_audio_from_video, convert_audio_format
from .document_processing import PDFTextExtractor
from .translation import Translator
from .summarization import Summarizer
from .subtitle import SubtitleGenerator
from .simple_text_processor import SimpleTextProcessor

logger = get_logger("media_to_text.processor")

class MediaProcessor:
    """
    Central processor for media-to-text conversion.
    Coordinates all core modules for a complete processing pipeline.
    """
    
    def __init__(
        self,
        model_size: str = 'base',
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        device: str = 'cpu',
        debug: bool = False
    ):
        """
        Initialize the media processor with specified settings.
        
        Args:
            model_size: Size of the Whisper model
            language: Source language code (or None for auto-detection)
            target_language: Target language code for translation (if None, no translation)
            device: Device to use for inference ('cpu' or 'cuda')
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = get_logger("media_to_text.processor", debug_mode=True)
        
        self.model_size = model_size
        self.source_language = language
        self.target_language = target_language
        self.device = device
        self.debug = debug
        
        logger.info(f"Initializing MediaProcessor with model_size={model_size}, "
                  f"language={language}, target_language={target_language}, device={device}")
        
        # Lazily loaded components
        self._transcriber = None
        self._document_processor = None
        self._translator = None
        self._summarizer = None
        self._subtitle_generator = None
        
        # Fallback flags - Use simple summarization by default to avoid segmentation faults
        self._use_simple_summarization = True
        self._use_simple_text_processor = SimpleTextProcessor() # Always available
        
        # Set OpenMP environment variable to prevent conflicts
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    @property
    def transcriber(self):
        """Lazy-loaded transcriber instance"""
        if self._transcriber is None:
            logger.info("Initializing transcriber")
            self._transcriber = Transcriber(
                model_size=self.model_size,
                language=self.source_language,
                device=self.device,
                debug=self.debug
            )
        return self._transcriber
    
    def _get_translator(self) -> Optional[Translator]:
        """Lazily load the translator."""
        if self._translator is None and self.target_language:
            logger.info(f"Initializing translator for {self.source_language} -> {self.target_language}")
            try:
                self._translator = Translator(
                    source_lang=self.source_language,
                    target_lang=self.target_language,
                    device=self.device,
                    debug=self.debug
                )
            except Exception as e:
                logger.error(f"Error initializing translator: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("Translation functionality disabled due to error")
                return None
                
        return self._translator
    
    def _get_summarizer(self) -> Union[Summarizer, SimpleTextProcessor]:
        """Lazily load the summarizer."""
        if self._use_simple_summarization:
            return self._use_simple_text_processor
            
        if self._summarizer is None:
            logger.info("Initializing summarizer")
            try:
                self._summarizer = Summarizer(device=self.device, debug=self.debug)
            except Exception as e:
                logger.error(f"Error initializing summarizer: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("Falling back to simple text-based summarization")
                self._use_simple_summarization = True
                return self._use_simple_text_processor
                
        return self._summarizer
    
    @property
    def pdf_extractor(self):
        """Lazy-loaded PDF extractor instance"""
        if self._document_processor is None:
            logger.info("Initializing PDF extractor")
            self._document_processor = PDFTextExtractor(debug=self.debug)
        return self._document_processor
    
    @property
    def subtitle_generator(self):
        """Lazy-loaded subtitle generator instance"""
        if self._subtitle_generator is None:
            logger.info("Initializing subtitle generator")
            self._subtitle_generator = SubtitleGenerator(debug=self.debug)
        return self._subtitle_generator
    
    def process_audio(
        self,
        audio_path: str,
        output_dir: str,
        translate: bool = False,
        summarize: bool = False,
        generate_subtitles: bool = False,
        subtitle_format: str = 'srt'
    ) -> Dict[str, str]:
        """
        Process an audio file: transcribe, optionally translate, summarize, and generate subtitles.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory for output files
            translate: Whether to translate the transcription
            summarize: Whether to generate a summary
            generate_subtitles: Whether to generate subtitle files
            subtitle_format: Format for subtitle files ('srt' or 'vtt')
            
        Returns:
            Dictionary mapping output types to file paths
        """
        logger.info(f"Processing audio file: {audio_path}")
        output_files = {}
        
        # Transcribe audio
        segments, info = self.transcriber.transcribe(audio_path)
        
        # If language was auto-detected, update the source language
        if not self.source_language and info.language:
            self.source_language = info.language
            logger.info(f"Auto-detected language: {self.source_language}")
        
        # Write transcription to file
        transcription_path = get_output_path(audio_path, output_dir, "_transcript", "txt")
        with open(transcription_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n")
        output_files['transcription'] = transcription_path
        
        # Translate if requested
        if translate and self.target_language:
            translated_segments = self._get_translator().translate_segments(segments)
            
            # Write translated transcription to file
            translation_path = get_output_path(
                audio_path, output_dir, f"_transcript_{self.target_language}", "txt"
            )
            with open(translation_path, 'w', encoding='utf-8') as f:
                for segment in translated_segments:
                    f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] "
                          f"{segment.get('translated_text', '')}\n")
            output_files['translation'] = translation_path
            
            # Update segments for subtitle generation
            if generate_subtitles:
                segments = translated_segments
        
        # Generate subtitles if requested
        if generate_subtitles:
            # Optimize segments for subtitles
            subtitle_segments = self.subtitle_generator.split_long_segments(segments)
            
            # Generate subtitle file
            subtitle_path = get_output_path(audio_path, output_dir, "_subtitles", subtitle_format)
            translated_key = 'translated_text' if translate and self.target_language else None
            
            self.subtitle_generator.generate_subtitle_file(
                subtitle_segments, subtitle_path, subtitle_format, translated_key
            )
            output_files['subtitles'] = subtitle_path
        
        # Generate summary if requested
        if summarize:
            summarizer = self._get_summarizer()
            if summarizer is not None:
                summary = summarizer.summarize_segments(segments)
                
                # Write summary to file
                summary_path = get_output_path(audio_path, output_dir, "_summary", "txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary['summary'])
                output_files['summary'] = summary_path
                
                # Translate summary if needed
                if translate and self.target_language:
                    translator = self._get_translator()
                    if translator is not None:
                        translated_summary = translator.translate(summary['summary'])
                        trans_summary_path = get_output_path(
                            audio_path, output_dir, f"_summary_{self.target_language}", "txt"
                        )
                        with open(trans_summary_path, 'w', encoding='utf-8') as f:
                            f.write(translated_summary)
                        output_files['translated_summary'] = trans_summary_path
        
        return output_files
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        translate: bool = False,
        summarize: bool = False,
        generate_subtitles: bool = False,
        subtitle_format: str = 'srt'
    ) -> Dict[str, str]:
        """
        Process a video file: extract audio, transcribe, optionally translate,
        summarize, and generate subtitles.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory for output files
            translate: Whether to translate the transcription
            summarize: Whether to generate a summary
            generate_subtitles: Whether to generate subtitle files
            subtitle_format: Format for subtitle files ('srt' or 'vtt')
            
        Returns:
            Dictionary mapping output types to file paths
        """
        logger.info(f"Processing video file: {video_path}")
        
        # Extract audio from video
        audio_path = get_output_path(video_path, output_dir, "_audio", "wav")
        extract_audio_from_video(video_path, audio_path)
        
        # Process the extracted audio
        output_files = self.process_audio(
            audio_path, 
            output_dir, 
            translate=translate,
            summarize=summarize,
            generate_subtitles=generate_subtitles,
            subtitle_format=subtitle_format
        )
        
        # Add extracted audio to output files
        output_files['extracted_audio'] = audio_path
        
        return output_files
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        translate: bool = False,
        summarize: bool = False
    ) -> Dict[str, str]:
        """
        Process a PDF file: extract text, optionally translate and summarize.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory for output files
            translate: Whether to translate the extracted text
            summarize: Whether to generate a summary
            
        Returns:
            Dictionary mapping output types to file paths
        """
        logger.info(f"Processing PDF file: {pdf_path}")
        output_files = {}
        
        # Extract text from PDF
        text_path = get_output_path(pdf_path, output_dir, "_text", "txt")
        self.pdf_extractor.extract_text_to_file(pdf_path, text_path)
        output_files['extracted_text'] = text_path
        
        # Read extracted text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Translate if requested
        if translate and self.target_language:
            translator = self._get_translator()
            if translator is not None:
                translation_path = get_output_path(
                    pdf_path, output_dir, f"_text_{self.target_language}", "txt"
                )
                translator.translate_file(text_path, translation_path)
                output_files['translation'] = translation_path
        
        # Generate summary if requested
        if summarize:
            summarizer = self._get_summarizer()
            if summarizer is not None:
                summary_path = get_output_path(pdf_path, output_dir, "_summary", "txt")
                if self._use_simple_summarization:
                    # Use simple text-based summarization
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    summary = summarizer.simple_summarize(text_content)
                    output_files["summary"] = summarizer.write_text_to_file(summary, summary_path)
                    logger.info(f"Simple summary written to {output_files['summary']}")
                else:
                    # Use ML-based summarization
                    output_files["summary"] = summarizer.summarize_file(text_path, summary_path)
                    logger.info(f"Summary written to {output_files['summary']}")
                
                # Translate summary if needed
                if translate and self.target_language:
                    translator = self._get_translator()
                    if translator is not None:
                        trans_summary_path = get_output_path(
                            pdf_path, output_dir, f"_summary_{self.target_language}", "txt"
                        )
                        translated_summary = translator.translate(output_files["summary"])
                        
                        with open(trans_summary_path, 'w', encoding='utf-8') as f:
                            f.write(translated_summary)
                        output_files['translated_summary'] = trans_summary_path
        
        return output_files
    
    def process_text(
        self,
        text_path: str,
        output_dir: str,
        translate: bool = False,
        summarize: bool = False
    ) -> Dict[str, str]:
        """
        Process a text file: optionally translate and summarize.
        
        Args:
            text_path: Path to the text file
            output_dir: Directory for output files
            translate: Whether to translate the text
            summarize: Whether to generate a summary
            
        Returns:
            Dictionary mapping output types to file paths
        """
        logger.info(f"Processing text file: {text_path}")
        output_files = {}
        
        # Read text file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Translate if requested
        if translate and self.target_language:
            translator = self._get_translator()
            if translator is not None:
                translation_path = get_output_path(
                    text_path, output_dir, f"_translated_{self.target_language}", "txt"
                )
                translator.translate_file(text_path, translation_path)
                output_files['translation'] = translation_path
        
        # Generate summary if requested
        if summarize:
            summarizer = self._get_summarizer()
            if summarizer is not None:
                summary_path = get_output_path(text_path, output_dir, "_summary", "txt")
                if self._use_simple_summarization:
                    # Use simple text-based summarization
                    summary = summarizer.simple_summarize(text)
                    output_files["summary"] = summarizer.write_text_to_file(summary, summary_path)
                    logger.info(f"Simple summary written to {output_files['summary']}")
                else:
                    # Use ML-based summarization
                    output_files["summary"] = summarizer.summarize_file(text_path, summary_path)
                    logger.info(f"Summary written to {output_files['summary']}")
                
                # Translate summary if needed
                if translate and self.target_language:
                    translator = self._get_translator()
                    if translator is not None:
                        trans_summary_path = get_output_path(
                            text_path, output_dir, f"_summary_{self.target_language}", "txt"
                        )
                        translated_summary = translator.translate(output_files["summary"])
                        
                        with open(trans_summary_path, 'w', encoding='utf-8') as f:
                            f.write(translated_summary)
                        output_files['translated_summary'] = trans_summary_path
        
        return output_files
    
    def process_file(
        self,
        file_path: str,
        output_dir: str,
        translate: bool = False,
        summarize: bool = False,
        generate_subtitles: bool = False,
        subtitle_format: str = 'srt'
    ) -> Dict[str, str]:
        """
        Process a file based on its type.
        
        Args:
            file_path: Path to the file
            output_dir: Directory for output files
            translate: Whether to translate the output
            summarize: Whether to generate a summary
            generate_subtitles: Whether to generate subtitle files
            subtitle_format: Format for subtitle files ('srt' or 'vtt')
            
        Returns:
            Dictionary mapping output types to file paths
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect file type
        file_type = detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type} for {file_path}")
        
        # Process according to file type
        if file_type == 'audio':
            return self.process_audio(
                file_path, output_dir, translate, summarize, generate_subtitles, subtitle_format
            )
        elif file_type == 'video':
            return self.process_video(
                file_path, output_dir, translate, summarize, generate_subtitles, subtitle_format
            )
        elif file_type == 'pdf':
            return self.process_pdf(file_path, output_dir, translate, summarize)
        elif file_type == 'text':
            return self.process_text(file_path, output_dir, translate, summarize)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def process_batch(
        self,
        file_paths: List[str],
        output_dir: str,
        translate: bool = False,
        summarize: bool = False,
        generate_subtitles: bool = False,
        subtitle_format: str = 'srt'
    ) -> Dict[str, Dict[str, str]]:
        """
        Process multiple files in batch mode with progress bar.
        
        Args:
            file_paths: List of file paths to process
            output_dir: Directory for output files
            translate: Whether to translate the output
            summarize: Whether to generate a summary
            generate_subtitles: Whether to generate subtitle files
            subtitle_format: Format for subtitle files ('srt' or 'vtt')
            
        Returns:
            Dictionary mapping file paths to their output file dictionaries
        """
        results = {}
        
        # Process each file with progress bar
        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            try:
                file_output_dir = os.path.join(output_dir, Path(file_path).stem)
                file_results = self.process_file(
                    file_path, file_output_dir, translate, summarize, generate_subtitles, subtitle_format
                )
                results[file_path] = file_results
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results[file_path] = {"error": str(e)}
        
        return results
