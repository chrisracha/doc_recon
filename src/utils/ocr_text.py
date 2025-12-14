"""
Text OCR module for document reconstruction.

Provides:
- Text extraction from image regions
- Multi-engine support (Tesseract, PaddleOCR, EasyOCR)
- Confidence scoring
- Post-processing (hyphenation fix, line merge)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WordResult:
    """OCR result for a single word."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


@dataclass
class LineResult:
    """OCR result for a line of text."""
    text: str
    confidence: float
    words: List[WordResult] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class OCRResult:
    """Complete OCR result for a text region."""
    text: str
    confidence: float
    lines: List[LineResult] = field(default_factory=list)
    raw_text: Optional[str] = None  # Before post-processing
    engine_used: str = ""
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.80
    
    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.65
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine_used,
            "alternatives": self.alternatives,
            "metadata": self.metadata
        }


# ============================================================================
# Text OCR Base Class
# ============================================================================

class TextOCR:
    """
    Main text OCR interface.
    
    Supports multiple engines with fallback:
    - Tesseract (baseline, always available)
    - PaddleOCR (better for degraded images)
    - EasyOCR (good for multiple languages)
    """
    
    def __init__(
        self,
        primary_engine: str = "tesseract",
        secondary_engine: Optional[str] = None,
        language: str = "eng",
        use_gpu: bool = False,
        confidence_threshold: float = 0.65
    ):
        self.primary_engine = primary_engine
        self.secondary_engine = secondary_engine
        self.language = language
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        
        self._engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines."""
        # Primary engine
        try:
            self._engines[self.primary_engine] = self._create_engine(self.primary_engine)
            logger.info(f"Initialized primary OCR engine: {self.primary_engine}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.primary_engine}: {e}")
            # Try fallback to tesseract
            if self.primary_engine != "tesseract":
                self._engines["tesseract"] = self._create_engine("tesseract")
                self.primary_engine = "tesseract"
        
        # Secondary engine (optional)
        if self.secondary_engine:
            try:
                self._engines[self.secondary_engine] = self._create_engine(self.secondary_engine)
                logger.info(f"Initialized secondary OCR engine: {self.secondary_engine}")
            except Exception as e:
                logger.warning(f"Failed to initialize secondary engine {self.secondary_engine}: {e}")
                self.secondary_engine = None
    
    def _create_engine(self, engine_name: str):
        """Create an OCR engine instance."""
        if engine_name == "tesseract":
            return TesseractEngine(language=self.language)
        elif engine_name == "paddleocr":
            return PaddleOCREngine(language=self.language, use_gpu=self.use_gpu)
        elif engine_name == "easyocr":
            return EasyOCREngine(language=self.language, use_gpu=self.use_gpu)
        else:
            raise ValueError(f"Unknown OCR engine: {engine_name}")
    
    def recognize(
        self,
        image: np.ndarray,
        use_secondary: bool = True,
        fix_hyphenation: bool = True,
        merge_lines: bool = True
    ) -> OCRResult:
        """
        Recognize text in an image region.
        
        Args:
            image: Input image (BGR or grayscale)
            use_secondary: Use secondary engine if primary confidence is low
            fix_hyphenation: Fix hyphenated words across lines
            merge_lines: Merge lines into paragraphs
            
        Returns:
            OCRResult with recognized text and confidence
        """
        # Run primary engine
        primary_result = self._engines[self.primary_engine].recognize(image)
        
        # If confidence is low and secondary engine is available, try it
        if (use_secondary and 
            self.secondary_engine and 
            primary_result.confidence < self.confidence_threshold):
            
            secondary_result = self._engines[self.secondary_engine].recognize(image)
            
            # Use whichever has higher confidence
            if secondary_result.confidence > primary_result.confidence:
                primary_result.alternatives.append(primary_result.text)
                primary_result = secondary_result
                primary_result.metadata["used_secondary"] = True
        
        # Post-processing
        primary_result.raw_text = primary_result.text
        
        if fix_hyphenation:
            primary_result.text = self._fix_hyphenation(primary_result.text)
        
        if merge_lines:
            primary_result.text = self._merge_lines(primary_result.text)
        
        return primary_result
    
    def recognize_blocks(
        self,
        image: np.ndarray,
        blocks: List[Any]
    ) -> List[OCRResult]:
        """
        Recognize text in multiple blocks.
        
        Args:
            image: Full page image
            blocks: List of Block objects with bboxes
            
        Returns:
            List of OCRResults for each block
        """
        results = []
        for block in blocks:
            # Crop block from image
            x1, y1, x2, y2 = block.bbox.to_tuple()
            block_image = image[y1:y2, x1:x2]
            
            if block_image.size == 0:
                results.append(OCRResult(text="", confidence=0.0))
                continue
            
            result = self.recognize(block_image)
            results.append(result)
        
        return results
    
    def _fix_hyphenation(self, text: str) -> str:
        """
        Fix hyphenated words that were split across lines.
        
        Example: "docu-\nment" -> "document"
        """
        # Pattern: word-\nword (hyphen at end of line followed by continuation)
        pattern = r'(\w+)-\s*\n\s*(\w+)'
        
        def replace_hyphen(match):
            word1 = match.group(1)
            word2 = match.group(2)
            # Check if it's likely a hyphenated compound word
            if word1.lower() in ['self', 'non', 'pre', 'post', 'anti', 'co', 're']:
                return f"{word1}-{word2}"
            return word1 + word2
        
        return re.sub(pattern, replace_hyphen, text)
    
    def _merge_lines(self, text: str) -> str:
        """
        Merge lines into proper paragraphs.
        
        Preserves paragraph breaks (blank lines) but removes
        artificial line breaks within paragraphs.
        """
        lines = text.split('\n')
        merged = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Blank line = paragraph break
                if current_para:
                    merged.append(' '.join(current_para))
                    current_para = []
                continue
            
            # Check if line ends with sentence-ending punctuation
            if current_para and not current_para[-1].rstrip().endswith(('.', '!', '?', ':')):
                current_para.append(line)
            else:
                if current_para:
                    merged.append(' '.join(current_para))
                current_para = [line]
        
        if current_para:
            merged.append(' '.join(current_para))
        
        return '\n\n'.join(merged)


# ============================================================================
# Tesseract Engine
# ============================================================================

class TesseractEngine:
    """OCR using Tesseract."""
    
    def __init__(
        self,
        language: str = "eng",
        config: str = "--oem 3 --psm 6"
    ):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            # Test that tesseract is installed
            pytesseract.get_tesseract_version()
            
        except Exception as e:
            raise ImportError(
                f"Tesseract not available: {e}\n"
                "Install with: pip install pytesseract\n"
                "Also install Tesseract: https://github.com/tesseract-ocr/tesseract"
            )
        
        self.language = language
        self.config = config
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small (helps OCR accuracy)
        h, w = gray.shape
        if h < 30:
            scale = 30.0 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding for better text contrast
        # This helps with scanned documents
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text using Tesseract."""
        import cv2
        
        # Preprocess image for better OCR
        processed = self._preprocess_for_ocr(image)
        
        # Get detailed data with confidence
        try:
            data = self.pytesseract.image_to_data(
                processed,
                lang=self.language,
                config=self.config,
                output_type=self.pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return OCRResult(text="", confidence=0.0, engine_used="tesseract")
        
        # Parse results
        lines = []
        current_line = []
        current_line_num = -1
        confidences = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            line_num = data['line_num'][i]
            
            if conf < 0:  # -1 means no valid confidence
                continue
            
            if text:
                word = WordResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=(
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                )
                
                if line_num != current_line_num:
                    if current_line:
                        line_text = ' '.join(w.text for w in current_line)
                        line_conf = np.mean([w.confidence for w in current_line])
                        lines.append(LineResult(
                            text=line_text,
                            confidence=line_conf,
                            words=current_line
                        ))
                    current_line = [word]
                    current_line_num = line_num
                else:
                    current_line.append(word)
                
                confidences.append(conf / 100.0)
        
        # Don't forget the last line
        if current_line:
            line_text = ' '.join(w.text for w in current_line)
            line_conf = np.mean([w.confidence for w in current_line])
            lines.append(LineResult(
                text=line_text,
                confidence=line_conf,
                words=current_line
            ))
        
        # Combine all text
        full_text = '\n'.join(line.text for line in lines)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            lines=lines,
            engine_used="tesseract"
        )


# ============================================================================
# PaddleOCR Engine
# ============================================================================

class PaddleOCREngine:
    """OCR using PaddleOCR."""
    
    def __init__(
        self,
        language: str = "en",
        use_gpu: bool = False
    ):
        try:
            from paddleocr import PaddleOCR
            import logging
            # Suppress PaddleOCR logging
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            
            # Map common language codes
            lang_map = {"eng": "en", "chi_sim": "ch", "chi_tra": "chinese_cht"}
            paddle_lang = lang_map.get(language, language)
            
            # Try new API first, fall back to old API
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=use_gpu
                )
            except TypeError:
                # Older version with show_log
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=use_gpu,
                    show_log=False
                )
        except ImportError:
            raise ImportError(
                "PaddleOCR not available. Install with: pip install paddleocr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
        
        self.language = language
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text using PaddleOCR."""
        try:
            result = self.ocr.ocr(image, cls=True)
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine_used="paddleocr")
        
        if not result or not result[0]:
            return OCRResult(text="", confidence=0.0, engine_used="paddleocr")
        
        lines = []
        confidences = []
        
        for line_data in result[0]:
            if len(line_data) >= 2:
                bbox_points = line_data[0]
                text, conf = line_data[1]
                
                # Convert polygon to bounding box
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                
                word = WordResult(text=text, confidence=conf, bbox=bbox)
                lines.append(LineResult(
                    text=text,
                    confidence=conf,
                    words=[word],
                    bbox=bbox
                ))
                confidences.append(conf)
        
        # Sort lines by vertical position
        lines.sort(key=lambda l: l.bbox[1] if l.bbox else 0)
        
        full_text = '\n'.join(line.text for line in lines)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            lines=lines,
            engine_used="paddleocr"
        )


# ============================================================================
# EasyOCR Engine
# ============================================================================

class EasyOCREngine:
    """OCR using EasyOCR."""
    
    def __init__(
        self,
        language: str = "en",
        use_gpu: bool = False
    ):
        try:
            import easyocr
            
            # Map language codes
            lang_map = {"eng": "en", "chi_sim": "ch_sim", "chi_tra": "ch_tra"}
            easy_lang = lang_map.get(language, language)
            
            self.reader = easyocr.Reader(
                [easy_lang],
                gpu=use_gpu,
                verbose=False
            )
        except ImportError:
            raise ImportError(
                "EasyOCR not available. Install with: pip install easyocr"
            )
        
        self.language = language
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text using EasyOCR."""
        try:
            result = self.reader.readtext(image)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return OCRResult(text="", confidence=0.0, engine_used="easyocr")
        
        lines = []
        confidences = []
        
        for detection in result:
            bbox_points, text, conf = detection
            
            # Convert polygon to bounding box
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            
            word = WordResult(text=text, confidence=conf, bbox=bbox)
            lines.append(LineResult(
                text=text,
                confidence=conf,
                words=[word],
                bbox=bbox
            ))
            confidences.append(conf)
        
        # Sort lines by vertical position
        lines.sort(key=lambda l: l.bbox[1] if l.bbox else 0)
        
        full_text = '\n'.join(line.text for line in lines)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            lines=lines,
            engine_used="easyocr"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def detect_text_orientation(image: np.ndarray) -> int:
    """
    Detect text orientation (0, 90, 180, 270 degrees).
    
    Returns:
        Rotation angle needed to correct orientation
    """
    try:
        import pytesseract
        
        osd = pytesseract.image_to_osd(image)
        angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
        return angle
    except Exception as e:
        logger.warning(f"Could not detect orientation: {e}")
        return 0


def estimate_font_size(image: np.ndarray) -> Optional[float]:
    """
    Estimate the dominant font size in an image.
    
    Returns:
        Estimated font size in pixels, or None if detection fails
    """
    import cv2
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect heights of contours (likely characters)
    heights = []
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        if 5 < h < 200 and 2 < w < 200:  # Filter reasonable character sizes
            heights.append(h)
    
    if heights:
        return float(np.median(heights))
    return None


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
        
        # Create OCR engine
        ocr = TextOCR(
            primary_engine="tesseract",
            secondary_engine=None,
            language="eng"
        )
        
        # Recognize text
        result = ocr.recognize(image)
        
        print(f"Engine used: {result.engine_used}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Lines found: {len(result.lines)}")
        print("\n--- Text ---")
        print(result.text)
    else:
        print("Usage: python ocr_text.py <image_path>")

