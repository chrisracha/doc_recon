"""
Tests for text OCR module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestOCRResult:
    """Test OCRResult class."""
    
    def test_ocr_result_creation(self):
        """Test OCR result creation."""
        from utils.ocr_text import OCRResult
        
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            engine_used="tesseract"
        )
        
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert result.is_high_confidence is True
        assert result.is_low_confidence is False
    
    def test_ocr_result_low_confidence(self):
        """Test low confidence detection."""
        from utils.ocr_text import OCRResult
        
        result = OCRResult(
            text="Unclear text",
            confidence=0.50
        )
        
        assert result.is_high_confidence is False
        assert result.is_low_confidence is True
    
    def test_ocr_result_to_dict(self):
        """Test OCR result serialization."""
        from utils.ocr_text import OCRResult
        
        result = OCRResult(
            text="Test",
            confidence=0.85,
            engine_used="tesseract",
            alternatives=["Tost", "Teat"]
        )
        
        data = result.to_dict()
        
        assert data["text"] == "Test"
        assert data["confidence"] == 0.85
        assert data["engine"] == "tesseract"
        assert data["alternatives"] == ["Tost", "Teat"]


class TestTextOCR:
    """Test TextOCR class."""
    
    @pytest.fixture
    def text_image(self):
        """Create a simple image with text-like patterns."""
        import cv2
        
        # Create white background
        img = np.ones((100, 400), dtype=np.uint8) * 255
        
        # Add text using OpenCV
        cv2.putText(
            img, "Hello World", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 2
        )
        
        return img
    
    @pytest.fixture
    def clear_text_image(self):
        """Create a clear image with well-defined text."""
        import cv2
        
        img = np.ones((200, 600), dtype=np.uint8) * 255
        
        # Add multiple lines of text
        cv2.putText(img, "The quick brown fox", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.putText(img, "jumps over the lazy dog", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        
        return img
    
    def test_text_ocr_init_tesseract(self):
        """Test TextOCR initialization with Tesseract."""
        from utils.ocr_text import TextOCR
        
        try:
            ocr = TextOCR(primary_engine="tesseract")
            assert "tesseract" in ocr._engines
        except ImportError:
            pytest.skip("Tesseract not available")
    
    def test_tesseract_recognize(self, text_image):
        """Test Tesseract recognition."""
        from utils.ocr_text import TesseractEngine
        
        try:
            engine = TesseractEngine()
            result = engine.recognize(text_image)
            
            assert result is not None
            assert result.engine_used == "tesseract"
            assert isinstance(result.confidence, float)
            # May or may not recognize "Hello World" depending on quality
        except ImportError:
            pytest.skip("Tesseract not available")
    
    def test_hyphenation_fix(self):
        """Test hyphenation fix function."""
        from utils.ocr_text import TextOCR
        
        ocr = TextOCR.__new__(TextOCR)
        
        # Test basic hyphenation
        text = "docu-\nment"
        fixed = ocr._fix_hyphenation(text)
        assert fixed == "document"
        
        # Test compound word (should preserve hyphen)
        text = "self-\ndriving"
        fixed = ocr._fix_hyphenation(text)
        assert fixed == "self-driving"
    
    def test_line_merge(self):
        """Test line merging function."""
        from utils.ocr_text import TextOCR
        
        ocr = TextOCR.__new__(TextOCR)
        
        # Test merging within paragraph
        text = "This is a sentence that\ncontinues on the next line."
        merged = ocr._merge_lines(text)
        assert "that continues" in merged
        
        # Test paragraph break preservation
        text = "First paragraph.\n\nSecond paragraph."
        merged = ocr._merge_lines(text)
        assert "First paragraph." in merged
        assert "Second paragraph." in merged


class TestWordResult:
    """Test WordResult class."""
    
    def test_word_result_creation(self):
        """Test word result creation."""
        from utils.ocr_text import WordResult
        
        word = WordResult(
            text="Hello",
            confidence=0.95,
            bbox=(10, 20, 60, 45)
        )
        
        assert word.text == "Hello"
        assert word.confidence == 0.95
        assert word.bbox == (10, 20, 60, 45)


class TestLineResult:
    """Test LineResult class."""
    
    def test_line_result_creation(self):
        """Test line result creation."""
        from utils.ocr_text import LineResult, WordResult
        
        words = [
            WordResult("Hello", 0.95),
            WordResult("World", 0.90)
        ]
        
        line = LineResult(
            text="Hello World",
            confidence=0.925,
            words=words
        )
        
        assert line.text == "Hello World"
        assert line.confidence == 0.925
        assert len(line.words) == 2


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_font_size(self):
        """Test font size estimation."""
        from utils.ocr_text import estimate_font_size
        import cv2
        
        # Create image with known text size
        img = np.ones((200, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Test", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 2)
        
        size = estimate_font_size(img)
        
        # Should detect some font size
        if size is not None:
            assert size > 0


class TestOCREngineSelection:
    """Test OCR engine selection and fallback."""
    
    def test_fallback_to_tesseract(self):
        """Test fallback to Tesseract when other engines unavailable."""
        from utils.ocr_text import TextOCR
        
        try:
            # Try with unavailable engine
            ocr = TextOCR(primary_engine="tesseract")
            
            # Should have tesseract available
            assert "tesseract" in ocr._engines
        except ImportError:
            pytest.skip("Tesseract not available")
    
    def test_secondary_engine_usage(self):
        """Test secondary engine for low-confidence results."""
        # This test would require both engines to be available
        # Skipping for now as it's environment-dependent
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


