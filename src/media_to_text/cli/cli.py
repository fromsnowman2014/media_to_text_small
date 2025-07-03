"""
Command-line interface for the media-to-text converter.
"""

import os
import sys
import argparse
import glob
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from tqdm import tqdm

from ..utils.logging_utils import get_logger, setup_logger
from ..utils.file_utils import detect_file_type, find_supported_files
from ..core.processor import MediaProcessor

logger = get_logger("media_to_text.cli")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Media to Text Converter - Convert audio, video, and documents to text"
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "-i", "--input", 
        required=True,
        help="Input file path or directory containing files to process"
    )
    input_group.add_argument(
        "-r", "--recursive", 
        action="store_true",
        help="Process files recursively if input is a directory"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save output files (default: ./output)"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Model size for transcription (default: base)"
    )
    proc_group.add_argument(
        "-l", "--language",
        help="Source language code (e.g., 'en', 'ko') or auto-detect if not specified"
    )
    proc_group.add_argument(
        "-t", "--translate",
        help="Translate to specified language code (e.g., 'en', 'ko')"
    )
    proc_group.add_argument(
        "-s", "--summarize",
        action="store_true",
        help="Generate summary of the content"
    )
    proc_group.add_argument(
        "--subtitles",
        action="store_true",
        help="Generate subtitles for audio/video files"
    )
    proc_group.add_argument(
        "--subtitle-format",
        choices=["srt", "vtt"],
        default="srt",
        help="Format for subtitle files (default: srt)"
    )
    
    # Advanced options
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for inference (default: cpu)"
    )
    adv_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    adv_group.add_argument(
        "--log-file",
        help="Path to log file (if not specified, logs only to console)"
    )
    
    return parser.parse_args()


def get_input_files(input_path: str, recursive: bool) -> List[str]:
    """
    Get list of files to process.
    
    Args:
        input_path: Path to input file or directory
        recursive: Whether to search recursively if input_path is a directory
        
    Returns:
        List of file paths
    """
    if os.path.isfile(input_path):
        return [input_path]
    
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, '**' if recursive else '*')
        files = []
        
        # Find all files in the directory
        for file_path in glob.glob(pattern, recursive=recursive):
            if os.path.isfile(file_path):
                file_type = detect_file_type(file_path)
                if file_type != 'unknown':
                    files.append(file_path)
        
        return files
    
    raise FileNotFoundError(f"Input path not found: {input_path}")


def print_results_summary(results: Dict[str, Dict[str, str]]):
    """
    Print summary of processing results.
    
    Args:
        results: Dictionary mapping file paths to their output file dictionaries
    """
    print("\nProcessing Summary:")
    print("-----------------")
    
    success_count = 0
    error_count = 0
    output_types = {}
    
    for file_path, file_results in results.items():
        file_name = os.path.basename(file_path)
        
        if "error" in file_results:
            print(f"❌ {file_name}: {file_results['error']}")
            error_count += 1
        else:
            print(f"✅ {file_name}")
            success_count += 1
            
            # Count output types
            for output_type in file_results:
                if output_type not in output_types:
                    output_types[output_type] = 0
                output_types[output_type] += 1
    
    print("\nStatistics:")
    print(f"- Total files processed: {success_count + error_count}")
    print(f"- Successful: {success_count}")
    print(f"- Failed: {error_count}")
    
    if output_types:
        print("\nOutput files generated:")
        for output_type, count in output_types.items():
            print(f"- {output_type}: {count}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    if args.log_file:
        logger = setup_logger("media_to_text", args.log_file, log_level)
    else:
        logger = setup_logger("media_to_text", level=log_level)
    
    try:
        # Get input files
        input_files = get_input_files(args.input, args.recursive)
        
        if not input_files:
            logger.error(f"No supported files found in {args.input}")
            sys.exit(1)
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Initialize processor
        processor = MediaProcessor(
            model_size=args.model,
            language=args.language,
            target_language=args.translate,
            device=args.device,
            debug=args.debug
        )
        
        # Process files
        results = processor.process_batch(
            input_files,
            args.output_dir,
            translate=bool(args.translate),
            summarize=args.summarize,
            generate_subtitles=args.subtitles,
            subtitle_format=args.subtitle_format
        )
        
        # Print summary
        print_results_summary(results)
        
        # Success
        logger.info("Processing completed")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=args.debug)
        return 1


if __name__ == "__main__":
    # Set OpenMP environment variable to prevent conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sys.exit(main())
