"""
Tests for the text_splitter module which provides semantic text chunking.
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.media_to_text.utils.text_splitter import split_text_into_chunks


class TestTextSplitter(unittest.TestCase):
    """Test cases for the text_splitter module."""

    def test_empty_text(self):
        """Test that empty text returns an empty list."""
        result = split_text_into_chunks("")
        self.assertEqual(result, [])

    def test_short_text(self):
        """Test that text shorter than max_chunk_size is returned as-is."""
        text = "This is a short text."
        result = split_text_into_chunks(text, max_chunk_size=100)
        self.assertEqual(result, [text])

    def test_paragraph_boundaries(self):
        """Test that text is split respecting paragraph boundaries."""
        # Create a text with two paragraphs
        para1 = "This is the first paragraph. It has multiple sentences. Each should be preserved."
        para2 = "This is the second paragraph. It also has multiple sentences. They should be kept together."
        text = f"{para1}\n\n{para2}"
        
        # Set max_chunk_size to just fit the first paragraph
        max_size = len(para1) + 5
        result = split_text_into_chunks(text, max_chunk_size=max_size, respect_paragraphs=True)
        
        # Verify that paragraphs are preserved (not split across chunks)
        para1_found = False
        para2_found = False
        
        for chunk in result:
            # Each paragraph should be completely contained in at least one chunk
            if para1 in chunk:
                para1_found = True
            if para2 in chunk:
                para2_found = True
            
            # No chunk should have partial paragraphs split at inappropriate boundaries
            partial1 = para1[:len(para1)//2]
            partial2 = para2[:len(para2)//2]
            if partial1 in chunk and para1 not in chunk:
                self.fail(f"Paragraph 1 was split inappropriately: {chunk}")
            if partial2 in chunk and para2 not in chunk:
                self.fail(f"Paragraph 2 was split inappropriately: {chunk}")
        
        self.assertTrue(para1_found, "First paragraph not found in any chunk")
        self.assertTrue(para2_found, "Second paragraph not found in any chunk")

    def test_sentence_boundaries(self):
        """Test that text is split respecting sentence boundaries."""
        # Create a text with multiple sentences
        sentence1 = "This is the first sentence."
        sentence2 = "This is the second sentence."
        sentence3 = "This is the third sentence that's longer than the previous ones."
        text = f"{sentence1} {sentence2} {sentence3}"
        
        # Set max_chunk_size to fit the first two sentences
        max_size = len(sentence1) + len(sentence2) + 5
        result = split_text_into_chunks(text, max_chunk_size=max_size, respect_sentences=True)
        
        # Verify that sentences are not split across chunks
        # Check that sentences appear intact in some chunk
        s1_found = s2_found = s3_found = False
        
        for chunk in result:
            if sentence1 in chunk:
                s1_found = True
            if sentence2 in chunk:
                s2_found = True
            if sentence3 in chunk:
                s3_found = True
                
            # No sentence should be split across chunks
            partial1 = sentence1[:len(sentence1)//2]
            if partial1 in chunk and sentence1 not in chunk:
                self.fail(f"Sentence 1 was split inappropriately: {chunk}")
                
            partial3 = sentence3[:len(sentence3)//2]
            if partial3 in chunk and sentence3 not in chunk:
                self.fail(f"Sentence 3 was split inappropriately: {chunk}")
        
        self.assertTrue(s1_found, "First sentence not found intact in any chunk")
        self.assertTrue(s2_found, "Second sentence not found intact in any chunk")
        self.assertTrue(s3_found, "Third sentence not found intact in any chunk")

    def test_very_long_sentence(self):
        """Test handling of sentences longer than max_chunk_size."""
        long_sentence = "This is a very long sentence that needs to be split " * 10
        result = split_text_into_chunks(long_sentence, max_chunk_size=100, respect_sentences=True)
        
        # Should be split into multiple chunks
        self.assertTrue(len(result) > 1)
        
        # Verify that the chunks contain all the original words
        original_words = set(long_sentence.split())
        result_words = set()
        for chunk in result:
            result_words.update(chunk.split())
        
        # All original words should be present in the result
        self.assertTrue(original_words.issubset(result_words), 
                        f"Missing words: {original_words - result_words}")
        
        # Check that each chunk doesn't exceed max size
        for chunk in result:
            self.assertTrue(len(chunk) <= 100, 
                          f"Chunk exceeds max size: {len(chunk)} > 100")
        
        # The core phrase should appear in the results
        core_phrase = "very long sentence"
        phrase_found = any(core_phrase in chunk for chunk in result)
        self.assertTrue(phrase_found, f"Core phrase '{core_phrase}' not found in any chunk")

    def test_overlap(self):
        """Test that chunks have the specified overlap."""
        # Create a text with clear word boundaries
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        
        # Split with a small max_chunk_size and overlap
        result = split_text_into_chunks(text, max_chunk_size=20, overlap=5)
        
        # Check that the second chunk starts with some text from the end of the first chunk
        if len(result) >= 2:
            overlap_text = result[0][-5:]
            self.assertTrue(result[1].startswith(overlap_text) or 
                          any(word in result[1][:15] for word in result[0][-15:].split()))


if __name__ == "__main__":
    unittest.main()
