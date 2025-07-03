"""
Document processing utilities for text extraction from PDF files.
"""

import os
import re
from typing import List, Dict, Optional, Any, Tuple

import pdfplumber

from ..utils.logging_utils import get_logger

logger = get_logger("media_to_text.document_processing")

class PDFTextExtractor:
    """
    Extracts text from PDF files with structure preservation.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the PDF text extractor.
        
        Args:
            debug: Enable debug logging
        """
        if debug:
            global logger
            logger = get_logger("media_to_text.document_processing", debug_mode=True)
    
    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file, preserving page structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and text content
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            # Open PDF file
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                
                # Process each page
                for i, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {i+1} of {len(pdf.pages)}")
                    
                    # Extract text with layout preservation
                    text = page.extract_text()
                    if text:
                        pages.append({
                            "page_num": i+1,
                            "content": text
                        })
                    
                logger.info(f"Extracted text from {len(pages)} pages")
                return pages
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_to_file(self, pdf_path: str, output_path: str) -> str:
        """
        Extract text from a PDF file and write it to a text file.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to write the extracted text
            
        Returns:
            Path to the output file
        """
        pages = self.extract_text(pdf_path)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the extracted text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in pages:
                f.write(f"--- Page {page['page_num']} ---\n\n")
                f.write(page['content'])
                f.write("\n\n")
        
        logger.info(f"Extracted text written to {output_path}")
        return output_path
    
    def extract_with_sections(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file, attempting to identify document sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing section title and content
        """
        pages = self.extract_text(pdf_path)
        
        # Combine all pages
        full_text = "\n".join([page["content"] for page in pages])
        
        # Attempt to identify sections using heuristics
        # This is a simple implementation and may need refinement for specific documents
        section_pattern = re.compile(r'(?:^|\n)(\d+(?:\.\d+)*[\.\s]+[A-Z][^\n]{2,60})(?:\n|$)')
        
        sections = []
        last_pos = 0
        last_title = "Introduction"
        
        # Find potential section headers
        for match in section_pattern.finditer(full_text):
            if last_pos > 0:  # Not the first section
                section_content = full_text[last_pos:match.start()].strip()
                sections.append({
                    "title": last_title,
                    "content": section_content
                })
            
            last_title = match.group(1).strip()
            last_pos = match.end()
        
        # Add the last section
        if last_pos < len(full_text):
            sections.append({
                "title": last_title,
                "content": full_text[last_pos:].strip()
            })
        
        return sections
