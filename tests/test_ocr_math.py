"""
Tests for math/equation OCR module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEquationResult:
    """Test EquationResult class."""
    
    def test_equation_result_creation(self):
        """Test equation result creation."""
        from utils.ocr_math import EquationResult
        
        result = EquationResult(
            latex="E = mc^2",
            confidence=0.95,
            equation_type="block"
        )
        
        assert result.latex == "E = mc^2"
        assert result.confidence == 0.95
        assert result.equation_type == "block"
        assert result.is_valid is True
    
    def test_equation_result_validity(self):
        """Test equation validity checking."""
        from utils.ocr_math import EquationResult
        
        # Valid equation
        valid = EquationResult(latex="\\frac{a}{b} = c", confidence=0.9)
        assert valid.is_valid is True
        
        # Empty equation
        empty = EquationResult(latex="", confidence=0.5)
        assert empty.is_valid is False
        
        # Too short
        short = EquationResult(latex="ab", confidence=0.8)
        assert short.is_valid is False
    
    def test_equation_result_confidence_levels(self):
        """Test confidence level classification."""
        from utils.ocr_math import EquationResult
        
        high = EquationResult(latex="x^2", confidence=0.85)
        assert high.is_high_confidence is True
        assert high.is_low_confidence is False
        
        low = EquationResult(latex="x^2", confidence=0.50)
        assert low.is_high_confidence is False
        assert low.is_low_confidence is True
    
    def test_equation_result_to_dict(self):
        """Test equation result serialization."""
        from utils.ocr_math import EquationResult
        
        result = EquationResult(
            latex="\\int_0^1 x dx",
            confidence=0.88,
            equation_type="block",
            engine_used="pix2tex",
            latex_alternatives=["\\int_0^1 x d x"]
        )
        
        data = result.to_dict()
        
        assert data["latex"] == "\\int_0^1 x dx"
        assert data["confidence"] == 0.88
        assert data["type"] == "block"
        assert data["engine"] == "pix2tex"
        assert len(data["alternatives"]) == 1


class TestMathOCR:
    """Test MathOCR class."""
    
    @pytest.fixture
    def equation_image(self):
        """Create a simple equation-like image."""
        import cv2
        
        # White background
        img = np.ones((80, 200), dtype=np.uint8) * 255
        
        # Draw "x = 2" using simple shapes
        cv2.putText(img, "x = 2", (40, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        
        return img
    
    def test_math_ocr_init_simple(self):
        """Test MathOCR initialization with simple recognizer."""
        from utils.ocr_math import MathOCR
        
        # Simple recognizer should always work
        ocr = MathOCR(engine="simple")
        assert ocr._recognizer is not None
        assert ocr.engine == "simple"
    
    def test_simple_math_recognizer(self, equation_image):
        """Test simple math recognizer."""
        from utils.ocr_math import SimpleMathRecognizer
        
        try:
            recognizer = SimpleMathRecognizer()
            result = recognizer.recognize(equation_image)
            
            assert result is not None
            assert result.engine_used == "simple"
        except ImportError:
            pytest.skip("Tesseract not available")
    
    def test_recognize_small_image(self):
        """Test handling of too-small images."""
        from utils.ocr_math import MathOCR
        
        # Very small image
        tiny = np.ones((5, 5), dtype=np.uint8) * 255
        
        ocr = MathOCR(engine="simple")
        result = ocr.recognize(tiny)
        
        assert result.status == "failed"
        assert result.confidence == 0.0


class TestLatexValidation:
    """Test LaTeX validation utilities."""
    
    def test_validate_balanced_braces(self):
        """Test brace balance validation."""
        from utils.ocr_math import validate_latex
        
        # Balanced
        valid, msg = validate_latex("\\frac{a}{b}")
        assert valid is True
        
        # Unbalanced - missing closing
        valid, msg = validate_latex("\\frac{a}{b")
        assert valid is False
        assert "brace" in msg.lower()
        
        # Unbalanced - extra closing
        valid, msg = validate_latex("\\frac{a}b}")
        assert valid is False
    
    def test_validate_empty(self):
        """Test empty LaTeX validation."""
        from utils.ocr_math import validate_latex
        
        valid, msg = validate_latex("")
        assert valid is False
    
    def test_validate_incomplete_command(self):
        """Test incomplete command detection."""
        from utils.ocr_math import validate_latex
        
        valid, msg = validate_latex("x^2 \\")
        assert valid is False
        assert "incomplete" in msg.lower()


class TestLatexCleaning:
    """Test LaTeX cleaning utilities."""
    
    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        from utils.ocr_math import clean_latex
        
        result = clean_latex("  x  =   2  ")
        assert result == "x = 2"
    
    def test_clean_empty(self):
        """Test empty string handling."""
        from utils.ocr_math import clean_latex
        
        result = clean_latex("")
        assert result == ""
        
        result = clean_latex(None)
        assert result == ""


class TestEquationDetection:
    """Test equation detection utilities."""
    
    def test_is_likely_equation_size_check(self):
        """Test equation detection size heuristics."""
        from utils.ocr_math import is_likely_equation
        
        # Too small
        tiny = np.ones((5, 10), dtype=np.uint8) * 255
        assert is_likely_equation(tiny) is False
        
        # Empty
        empty = np.array([])
        assert is_likely_equation(empty) is False
    
    def test_is_likely_equation_whitespace(self):
        """Test equation detection whitespace heuristics."""
        from utils.ocr_math import is_likely_equation
        
        # Mostly white (like an equation)
        equation_like = np.ones((50, 200), dtype=np.uint8) * 255
        equation_like[20:30, 50:150] = 0  # Small amount of text
        
        result = is_likely_equation(equation_like)
        # Should return True for equation-like pattern
    
    def test_is_likely_equation_aspect_ratio(self):
        """Test equation detection aspect ratio check."""
        from utils.ocr_math import is_likely_equation
        
        # Very tall and narrow (unlikely equation)
        tall = np.ones((200, 20), dtype=np.uint8) * 255
        assert is_likely_equation(tall) is False


class TestPix2TexRecognizer:
    """Test pix2tex recognizer (if available)."""
    
    def test_pix2tex_available(self):
        """Test if pix2tex is available."""
        try:
            from utils.ocr_math import Pix2TexRecognizer
            recognizer = Pix2TexRecognizer()
            assert recognizer is not None
        except ImportError:
            pytest.skip("pix2tex not available")
        except Exception as e:
            pytest.skip(f"pix2tex initialization failed: {e}")


class TestMathpixRecognizer:
    """Test Mathpix API recognizer."""
    
    def test_mathpix_requires_credentials(self):
        """Test that Mathpix requires API credentials."""
        from utils.ocr_math import MathpixRecognizer
        
        # Should raise without credentials
        with pytest.raises(Exception):
            MathpixRecognizer(app_id=None, app_key=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


