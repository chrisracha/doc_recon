"""
Math/Equation OCR module for document reconstruction.

Provides:
- Equation image to LaTeX conversion
- Support for pix2tex (local) and Mathpix (API)
- Inline and block equation detection
- Confidence scoring and alternatives
"""

import logging
import base64
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EquationResult:
    """Result of equation OCR."""
    latex: str
    confidence: float
    latex_alternatives: List[str] = field(default_factory=list)
    equation_type: str = "block"  # "block" or "inline"
    engine_used: str = ""
    status: str = "success"  # success, uncertain, failed
    image_path: Optional[str] = None  # Path to cropped equation image
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if the LaTeX result appears valid."""
        if not self.latex or len(self.latex) < 3:
            return False
        # Check for common LaTeX elements
        latex_indicators = ['\\', '^', '_', '{', '}', '=', '+', '-', '*', '/']
        return any(ind in self.latex for ind in latex_indicators)
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.80
    
    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.65
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latex": self.latex,
            "confidence": self.confidence,
            "alternatives": self.latex_alternatives,
            "type": self.equation_type,
            "engine": self.engine_used,
            "status": self.status,
            "image_path": self.image_path,
            "metadata": self.metadata
        }


# ============================================================================
# Math OCR Main Class
# ============================================================================

class MathOCR:
    """
    Main math/equation OCR interface.
    
    Supports:
    - pix2tex (local, PyTorch-based)
    - Mathpix API (optional, for degraded images)
    """
    
    def __init__(
        self,
        engine: str = "pix2tex",
        mathpix_app_id: Optional[str] = None,
        mathpix_app_key: Optional[str] = None,
        use_gpu: bool = False,
        confidence_threshold: float = 0.65
    ):
        self.engine = engine
        self.mathpix_app_id = mathpix_app_id
        self.mathpix_app_key = mathpix_app_key
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        
        self._recognizer = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the math OCR engine."""
        if self.engine == "pix2tex":
            try:
                self._recognizer = Pix2TexRecognizer(use_gpu=self.use_gpu)
                logger.info("Initialized pix2tex for math OCR")
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to initialize pix2tex: {e}")
                
                # Check if it's the albumentations compatibility issue
                if "std_range" in error_msg or "albumentations" in error_msg.lower():
                    logger.error(
                        "pix2tex compatibility issue detected!\n"
                        "Fix: pip install 'albumentations<1.4.0'\n"
                        "Or: pip uninstall albumentations && pip install 'albumentations==1.3.1'"
                    )
                
                # Fall back to Mathpix if available
                if self.mathpix_app_id and self.mathpix_app_key:
                    try:
                        self._recognizer = MathpixRecognizer(
                            self.mathpix_app_id,
                            self.mathpix_app_key
                        )
                        self.engine = "mathpix"
                        logger.info("Falling back to Mathpix API for math OCR")
                    except Exception:
                        pass
                
                # Final fallback to simple
                if self._recognizer is None:
                    self._recognizer = SimpleMathRecognizer()
                    self.engine = "simple"
                    logger.warning("Using simple math recognizer (limited capability)")
        
        elif self.engine == "mathpix":
            if not self.mathpix_app_id or not self.mathpix_app_key:
                raise ValueError("Mathpix requires app_id and app_key")
            self._recognizer = MathpixRecognizer(
                self.mathpix_app_id,
                self.mathpix_app_key
            )
            logger.info("Using Mathpix API for math OCR")
        
        else:
            raise ValueError(f"Unknown math OCR engine: {self.engine}")
    
    def recognize(
        self,
        image: np.ndarray,
        equation_type: str = "block",
        save_image: Optional[str] = None
    ) -> EquationResult:
        """
        Recognize equation from image.
        
        Args:
            image: Cropped equation image (BGR or grayscale)
            equation_type: "block" for display equations, "inline" for inline math
            save_image: Optional path to save the equation image
            
        Returns:
            EquationResult with LaTeX and confidence
        """
        import cv2
        
        # Validate image size
        h, w = image.shape[:2]
        if w < 10 or h < 5:
            return EquationResult(
                latex="",
                confidence=0.0,
                status="failed",
                metadata={"error": "Image too small"}
            )
        
        # Save image if requested
        image_path = None
        if save_image:
            cv2.imwrite(save_image, image)
            image_path = save_image
        
        # Run recognition
        result = self._recognizer.recognize(image)
        result.equation_type = equation_type
        result.engine_used = self.engine
        result.image_path = image_path
        
        # Validate and set status
        if result.confidence < self.confidence_threshold:
            result.status = "uncertain"
        elif not result.is_valid:
            result.status = "uncertain"
        
        return result
    
    def recognize_batch(
        self,
        images: List[np.ndarray],
        equation_types: Optional[List[str]] = None
    ) -> List[EquationResult]:
        """
        Recognize multiple equations.
        
        Args:
            images: List of equation images
            equation_types: Optional list of equation types
            
        Returns:
            List of EquationResults
        """
        if equation_types is None:
            equation_types = ["block"] * len(images)
        
        results = []
        for img, eq_type in zip(images, equation_types):
            result = self.recognize(img, equation_type=eq_type)
            results.append(result)
        
        return results


# ============================================================================
# Pix2Tex Recognizer
# ============================================================================

class Pix2TexRecognizer:
    """Equation recognition using pix2tex (LaTeX-OCR)."""
    
    def __init__(self, use_gpu: bool = False):
        try:
            from pix2tex.cli import LatexOCR
            
            try:
                self.model = LatexOCR()
                logger.info("Loaded pix2tex model")
            except (FileNotFoundError, OSError) as e:
                # Handle missing file errors (e.g., app.py in model/src)
                if "app.py" in str(e) or "No such file" in str(e):
                    logger.warning(f"pix2tex file error (may be harmless): {e}")
                    # Try to continue anyway - the model might still work
                    try:
                        self.model = LatexOCR()
                    except:
                        raise RuntimeError(
                            f"pix2tex installation issue: {e}\n"
                            "Try: pip uninstall pix2tex && pip install pix2tex"
                        )
                else:
                    raise
        except ImportError:
            raise ImportError(
                "pix2tex not available. Install with: pip install pix2tex"
            )
        except (ValueError, RuntimeError) as e:
            # Handle albumentations compatibility issues
            if "std_range" in str(e) or "albumentations" in str(e).lower():
                raise RuntimeError(
                    f"pix2tex compatibility error: {e}\n"
                    "Fix: pip install 'albumentations<1.4.0'"
                )
            raise RuntimeError(f"Failed to load pix2tex model: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pix2tex model: {e}")
        
        self.use_gpu = use_gpu
    
    def recognize(self, image: np.ndarray) -> EquationResult:
        """Recognize equation using pix2tex."""
        import cv2
        from PIL import Image
        
        try:
            # Convert to PIL Image (RGB)
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(rgb)
            
            # Run inference
            latex = self.model(pil_image)
            
            # pix2tex doesn't provide confidence, so we estimate based on length and validity
            confidence = self._estimate_confidence(latex)
            
            return EquationResult(
                latex=latex,
                confidence=confidence,
                engine_used="pix2tex"
            )
            
        except Exception as e:
            logger.error(f"pix2tex error: {e}")
            return EquationResult(
                latex="",
                confidence=0.0,
                status="failed",
                metadata={"error": str(e)}
            )
    
    def _estimate_confidence(self, latex: str) -> float:
        """Estimate confidence based on LaTeX quality."""
        if not latex:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on LaTeX characteristics
        if len(latex) > 10:
            confidence += 0.1
        
        # Check for balanced braces
        if latex.count('{') == latex.count('}'):
            confidence += 0.1
        else:
            confidence -= 0.2
        
        # Check for common math commands
        math_commands = ['\\frac', '\\int', '\\sum', '\\sqrt', '\\alpha', '\\beta']
        if any(cmd in latex for cmd in math_commands):
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)


# ============================================================================
# Mathpix Recognizer
# ============================================================================

class MathpixRecognizer:
    """Equation recognition using Mathpix API."""
    
    API_URL = "https://api.mathpix.com/v3/latex"
    
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
    
    def recognize(self, image: np.ndarray) -> EquationResult:
        """Recognize equation using Mathpix API."""
        import cv2
        
        try:
            # Encode image as base64
            _, buffer = cv2.imencode('.png', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare request
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-type": "application/json"
            }
            
            data = {
                "src": f"data:image/png;base64,{image_b64}",
                "formats": ["latex_styled", "latex_normal"],
                "include_line_data": True
            }
            
            # Send request
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract LaTeX
            latex = result.get("latex_styled", result.get("latex_normal", ""))
            confidence = result.get("confidence", 0.5)
            
            # Get alternatives
            alternatives = []
            if "latex_normal" in result and result["latex_normal"] != latex:
                alternatives.append(result["latex_normal"])
            
            return EquationResult(
                latex=latex,
                confidence=confidence,
                latex_alternatives=alternatives,
                engine_used="mathpix"
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Mathpix API error: {e}")
            return EquationResult(
                latex="",
                confidence=0.0,
                status="failed",
                metadata={"error": f"API error: {e}"}
            )
        except Exception as e:
            logger.error(f"Mathpix error: {e}")
            return EquationResult(
                latex="",
                confidence=0.0,
                status="failed",
                metadata={"error": str(e)}
            )


# ============================================================================
# Simple Math Recognizer (Fallback)
# ============================================================================

class SimpleMathRecognizer:
    """
    Simple math recognizer using Tesseract with math mode.
    
    This is a fallback when pix2tex is not available.
    Has limited capability for complex equations.
    """
    
    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except ImportError:
            raise ImportError("pytesseract required for simple math recognition")
    
    def recognize(self, image: np.ndarray) -> EquationResult:
        """Recognize simple math using Tesseract."""
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Invert if needed (white text on black)
        if np.mean(gray) < 128:
            gray = 255 - gray
        
        try:
            # Use tesseract with custom config for math
            text = self.pytesseract.image_to_string(
                gray,
                config="--psm 6 -c tessedit_char_whitelist=0123456789+-*/=()^_{}[].,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\\αβγδεζηθικλμνξπρστυφχψω"
            ).strip()
            
            # Convert to basic LaTeX
            latex = self._text_to_latex(text)
            
            # Confidence is low for this method
            confidence = 0.4 if latex else 0.0
            
            return EquationResult(
                latex=latex,
                confidence=confidence,
                engine_used="simple",
                status="uncertain" if confidence < 0.5 else "success"
            )
            
        except Exception as e:
            logger.error(f"Simple math recognizer error: {e}")
            return EquationResult(
                latex="",
                confidence=0.0,
                status="failed",
                metadata={"error": str(e)}
            )
    
    def _text_to_latex(self, text: str) -> str:
        """Convert OCR text to basic LaTeX."""
        if not text:
            return ""
        
        # Basic substitutions
        replacements = [
            ('*', ' \\times '),
            ('/', ' \\div '),
            ('sqrt', '\\sqrt'),
            ('alpha', '\\alpha'),
            ('beta', '\\beta'),
            ('gamma', '\\gamma'),
            ('delta', '\\delta'),
            ('pi', '\\pi'),
            ('sum', '\\sum'),
            ('int', '\\int'),
            ('inf', '\\infty'),
            ('>=', '\\geq'),
            ('<=', '\\leq'),
            ('!=', '\\neq'),
        ]
        
        latex = text
        for old, new in replacements:
            latex = latex.replace(old, new)
        
        return latex


# ============================================================================
# Utility Functions
# ============================================================================

def detect_equation_regions(
    image: np.ndarray,
    text_blocks: List[Any]
) -> List[Tuple[np.ndarray, str]]:
    """
    Detect equation regions in an image.
    
    Args:
        image: Full page image
        text_blocks: List of text blocks from layout detection
        
    Returns:
        List of (equation_image, equation_type) tuples
    """
    import cv2
    
    equations = []
    h, w = image.shape[:2]
    
    for block in text_blocks:
        # Check if block might be an equation
        block_img = block.crop_from_image(image)
        
        if is_likely_equation(block_img):
            eq_type = "block" if block.bbox.width > w * 0.3 else "inline"
            equations.append((block_img, eq_type))
    
    return equations


def is_likely_equation(image: np.ndarray) -> bool:
    """
    Heuristically determine if an image region is likely an equation.
    
    Args:
        image: Image region
        
    Returns:
        True if likely an equation
    """
    import cv2
    
    if image.size == 0:
        return False
    
    h, w = image.shape[:2]
    
    # Size heuristics
    if w < 20 or h < 10:
        return False
    
    # Aspect ratio (equations tend to be wider than tall)
    aspect = w / h
    if aspect < 0.3:  # Very tall and narrow = probably not equation
        return False
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Check white space ratio (equations have more white space)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary > 200) / binary.size
    
    # Equations typically have 60-90% white space
    if 0.5 < white_ratio < 0.95:
        return True
    
    return False


def validate_latex(latex: str) -> Tuple[bool, str]:
    """
    Validate LaTeX syntax.
    
    Args:
        latex: LaTeX string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not latex:
        return False, "Empty LaTeX string"
    
    # Check balanced braces
    brace_count = 0
    for char in latex:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        if brace_count < 0:
            return False, "Unbalanced braces"
    
    if brace_count != 0:
        return False, "Unbalanced braces"
    
    # Check for incomplete commands
    if latex.endswith('\\'):
        return False, "Incomplete command"
    
    return True, ""


def clean_latex(latex: str) -> str:
    """
    Clean and normalize LaTeX string.
    
    Args:
        latex: Raw LaTeX string
        
    Returns:
        Cleaned LaTeX string
    """
    import re
    
    if not latex:
        return ""
    
    # Remove leading/trailing whitespace
    latex = latex.strip()
    
    # Remove extra whitespace
    latex = re.sub(r'\s+', ' ', latex)
    
    # Normalize common variations
    latex = latex.replace('\\left(', '(')
    latex = latex.replace('\\right)', ')')
    latex = latex.replace('\\left[', '[')
    latex = latex.replace('\\right]', ']')
    
    return latex


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import cv2
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            sys.exit(1)
        
        # Create math OCR engine
        try:
            math_ocr = MathOCR(engine="pix2tex", use_gpu=False)
        except Exception as e:
            print(f"Warning: {e}")
            print("Using simple math recognizer")
            math_ocr = MathOCR(engine="simple")
        
        # Recognize equation
        result = math_ocr.recognize(image)
        
        print(f"Engine: {result.engine_used}")
        print(f"Status: {result.status}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"LaTeX: {result.latex}")
        
        if result.latex_alternatives:
            print(f"Alternatives: {result.latex_alternatives}")
    else:
        print("Usage: python ocr_math.py <equation_image>")

