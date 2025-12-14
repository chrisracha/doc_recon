"""
Layout detection module for document reconstruction.

Provides:
- Block detection (text, tables, figures, equations)
- Block classification
- Reading order resolution
- Multi-column detection

Supports:
- LayoutParser with Detectron2 (primary)
- PaddleOCR layout (alternative)
- Classical CV-based fallback
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Enums
# ============================================================================

class BlockType(Enum):
    """Types of document blocks."""
    TITLE = "title"
    AUTHORS = "authors"
    ABSTRACT = "abstract"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    EQUATION_BLOCK = "equation_block"
    EQUATION_INLINE = "equation_inline"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    REFERENCES = "references"
    FOOTER = "footer"
    HEADER = "header"
    LIST = "list"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)
    
    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> 'BoundingBox':
        return cls(x, y, x + w, y + h)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return not (
            self.x2 < other.x1 or self.x1 > other.x2 or
            self.y2 < other.y1 or self.y1 > other.y2
        )
    
    def intersection_area(self, other: 'BoundingBox') -> int:
        if not self.intersects(other):
            return 0
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0


@dataclass
class Block:
    """A detected block in the document."""
    bbox: BoundingBox
    block_type: BlockType
    confidence: float = 0.0
    text: Optional[str] = None
    reading_order: int = 0
    column_index: int = 0
    parent_id: Optional[str] = None
    block_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.block_id:
            import uuid
            self.block_id = str(uuid.uuid4())[:8]
    
    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract the block region from an image."""
        return image[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "type": self.block_type.value,
            "bbox": self.bbox.to_tuple(),
            "confidence": self.confidence,
            "text": self.text,
            "reading_order": self.reading_order,
            "column_index": self.column_index,
            "metadata": self.metadata
        }


@dataclass
class LayoutResult:
    """Result of layout detection."""
    blocks: List[Block]
    page_width: int
    page_height: int
    num_columns: int = 1
    confidence: float = 0.0
    method_used: str = ""
    debug_image: Optional[np.ndarray] = None


# ============================================================================
# Layout Detector Base Class
# ============================================================================

class LayoutDetector:
    """
    Main layout detection interface.
    
    Tries LayoutParser first, falls back to classical methods if unavailable.
    """
    
    def __init__(
        self,
        model_type: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False,
        fallback_to_classical: bool = True,
        layout_method: Optional[str] = None  # "auto", "pix2text", "layoutparser", "paddleocr", "classical"
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.fallback_to_classical = fallback_to_classical
        self.layout_method = layout_method or "auto"
        self._detector = None
        self._method = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Try to initialize the specified or best available detector."""
        # If specific method requested, try only that
        if self.layout_method != "auto":
            if self.layout_method == "pix2text":
                self._try_pix2text()
            elif self.layout_method == "layoutparser":
                self._try_layoutparser()
            elif self.layout_method == "paddleocr":
                self._try_paddleocr()
            elif self.layout_method == "classical":
                self._use_classical()
            
            if self._detector:
                return
        
        # Auto mode: try in order of preference
        # 1. LayoutParser (most reliable if available)
        if self._try_layoutparser():
            return
        
        # 2. PaddleOCR (good alternative)
        if self._try_paddleocr():
            return
        
        # 3. Pix2Text (good for equations/tables but can be slower)
        if self._try_pix2text():
            return
        
        # 4. Classical CV (always works)
        if self.fallback_to_classical:
            self._use_classical()
        else:
            raise RuntimeError("No layout detection method available")
    
    def _try_pix2text(self) -> bool:
        """Try to initialize Pix2Text."""
        try:
            self._detector = Pix2TextLayoutDetector(use_gpu=self.use_gpu)
            self._method = "pix2text"
            logger.info("Using Pix2Text for layout detection (automatic equation/table detection)")
            return True
        except ImportError as e:
            logger.debug(f"Pix2Text not available: {e}")
        except Exception as e:
            logger.debug(f"Failed to initialize Pix2Text: {e}")
        return False
    
    def _try_layoutparser(self) -> bool:
        """Try to initialize LayoutParser."""
        try:
            self._detector = LayoutParserDetector(
                model_type=self.model_type,
                confidence_threshold=self.confidence_threshold,
                use_gpu=self.use_gpu
            )
            self._method = "layoutparser"
            logger.info("Using LayoutParser for layout detection")
            return True
        except ImportError as e:
            logger.debug(f"LayoutParser not available: {e}")
        except Exception as e:
            logger.debug(f"Failed to initialize LayoutParser: {e}")
        return False
    
    def _try_paddleocr(self) -> bool:
        """Try to initialize PaddleOCR."""
        try:
            self._detector = PaddleLayoutDetector(use_gpu=self.use_gpu)
            self._method = "paddleocr"
            logger.info("Using PaddleOCR for layout detection")
            return True
        except ImportError as e:
            logger.debug(f"PaddleOCR not available: {e}")
        except Exception as e:
            logger.debug(f"Failed to initialize PaddleOCR: {e}")
        return False
    
    def _use_classical(self):
        """Use classical CV method."""
        self._detector = ClassicalLayoutDetector(min_block_area=100)
        self._method = "classical"
        logger.info("Using classical CV for layout detection")
    
    def detect(
        self,
        image: np.ndarray,
        debug: bool = False
    ) -> LayoutResult:
        """
        Detect layout blocks in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            debug: If True, include debug visualization
            
        Returns:
            LayoutResult with detected blocks
        """
        result = self._detector.detect(image, debug=debug)
        result.method_used = self._method
        
        # Post-process: assign reading order
        result.blocks = self._assign_reading_order(result.blocks, result.num_columns)
        
        # Classify blocks that may need refinement
        result.blocks = self._refine_block_types(result.blocks, image)
        
        return result
    
    def _assign_reading_order(
        self,
        blocks: List[Block],
        num_columns: int = 1
    ) -> List[Block]:
        """Assign reading order to blocks."""
        if not blocks:
            return blocks
        
        # Get page width from blocks
        page_width = max(b.bbox.x2 for b in blocks)
        
        # Detect columns
        if num_columns == 1:
            num_columns = self._detect_num_columns(blocks, page_width)
        
        column_width = page_width / num_columns
        
        # Assign column indices
        for block in blocks:
            center_x = block.bbox.center[0]
            block.column_index = min(int(center_x / column_width), num_columns - 1)
        
        # Sort by column, then by vertical position
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (b.column_index, b.bbox.y1)
        )
        
        # Assign reading order
        for i, block in enumerate(sorted_blocks):
            block.reading_order = i
        
        return sorted_blocks
    
    def _detect_num_columns(
        self,
        blocks: List[Block],
        page_width: int
    ) -> int:
        """Detect the number of columns in the layout."""
        if not blocks:
            return 1
        
        # Get x-coordinates of block centers
        centers = [b.bbox.center[0] for b in blocks]
        
        # Simple heuristic: if there's a clear gap in the middle, it's 2 columns
        mid = page_width / 2
        left_count = sum(1 for c in centers if c < mid * 0.8)
        right_count = sum(1 for c in centers if c > mid * 1.2)
        
        if left_count >= 3 and right_count >= 3:
            # Check if the gap is significant
            left_max = max(b.bbox.x2 for b in blocks if b.bbox.center[0] < mid * 0.8)
            right_min = min(b.bbox.x1 for b in blocks if b.bbox.center[0] > mid * 1.2)
            
            if right_min - left_max > page_width * 0.05:
                return 2
        
        return 1
    
    def _refine_block_types(
        self,
        blocks: List[Block],
        image: np.ndarray
    ) -> List[Block]:
        """Refine block type classifications."""
        page_height = image.shape[0]
        
        for block in blocks:
            # Header detection (top 10% of page)
            if block.bbox.y1 < page_height * 0.1:
                if block.block_type == BlockType.PARAGRAPH:
                    block.block_type = BlockType.HEADER
            
            # Footer detection (bottom 10% of page)
            if block.bbox.y2 > page_height * 0.9:
                if block.block_type == BlockType.PARAGRAPH:
                    block.block_type = BlockType.FOOTER
            
            # Title detection (first block, large, centered)
            if block.reading_order == 0:
                if block.bbox.height > page_height * 0.02:
                    block.block_type = BlockType.TITLE
        
        return blocks


# ============================================================================
# Pix2Text Layout Implementation (Automatic Equation/Table Detection)
# ============================================================================

class Pix2TextLayoutDetector:
    """
    Layout detection using Pix2Text.
    
    Pix2Text automatically detects:
    - Equations (and converts to LaTeX)
    - Tables (and extracts structure)
    - Text blocks
    - Layout regions
    
    This is a great all-in-one solution for academic documents!
    """
    
    LABEL_MAP = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.TITLE,
        "figure": BlockType.FIGURE,
        "table": BlockType.TABLE,
        "equation": BlockType.EQUATION_BLOCK,
        "equation-inline": BlockType.EQUATION_INLINE,
    }
    
    def __init__(self, use_gpu: bool = False):
        try:
            from pix2text import Pix2Text
            import logging
            logging.getLogger('pix2text').setLevel(logging.WARNING)
            
            # Initialize Pix2Text
            # It automatically handles equations and tables
            self.p2t = Pix2Text(device='cuda' if use_gpu else 'cpu')
            logger.info("Pix2Text initialized - automatic equation/table detection enabled")
        except ImportError:
            raise ImportError(
                "Pix2Text not available. Install with: pip install pix2text"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pix2Text: {e}")
    
    def detect(self, image: np.ndarray, debug: bool = False) -> LayoutResult:
        """Detect layout using Pix2Text."""
        import cv2
        from PIL import Image
        import tempfile
        from pathlib import Path
        
        # Convert to PIL Image (RGB)
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_image)
        
        # Run Pix2Text - it automatically detects equations and tables!
        # Pix2Text can work with PIL Image or file path
        result = None
        try:
            # Try recognize_text_page first (returns Page object with chunks)
            if hasattr(self.p2t, 'recognize_text_page'):
                result = self.p2t.recognize_text_page(pil_image)
                logger.debug(f"Pix2Text recognize_text_page() result type: {type(result)}")
            else:
                # Fallback to recognize method
                result = self.p2t.recognize(pil_image)
                logger.debug(f"Pix2Text recognize() result type: {type(result)}")
        except Exception as e:
            logger.error(f"Pix2Text recognition error: {e}", exc_info=True)
            # Return empty result but don't crash
            return LayoutResult(
                blocks=[],
                page_width=image.shape[1],
                page_height=image.shape[0],
                confidence=0.0,
                method_used="pix2text"
            )
        
        if result is None:
            logger.error("Pix2Text returned None")
            return LayoutResult(
                blocks=[],
                page_width=image.shape[1],
                page_height=image.shape[0],
                confidence=0.0,
                method_used="pix2text"
            )
        
        # Parse Pix2Text output
        blocks = []
        h, w = image.shape[:2]
        
        # Pix2Text recognize_text_page returns a Page object with .chunks attribute
        # Each chunk has: type, bbox, text, etc.
        chunks = []
        
        if hasattr(result, 'chunks'):
            chunks = result.chunks
            logger.debug(f"Found {len(chunks)} chunks from Pix2Text")
        elif hasattr(result, 'elements'):
            # Alternative attribute name
            chunks = result.elements
            logger.debug(f"Found {len(chunks)} elements from Pix2Text")
        elif isinstance(result, dict):
            chunks = result.get('chunks', result.get('elements', []))
            logger.debug(f"Found {len(chunks)} chunks from dict result")
        elif isinstance(result, list):
            chunks = result
            logger.debug(f"Result is a list with {len(chunks)} items")
        else:
            # Result might be a string (markdown) or Page object
            logger.warning(f"Pix2Text returned unexpected format: {type(result)}")
            logger.debug(f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'N/A'}")
            
            # Try to get chunks from Page object
            if hasattr(result, '__dict__'):
                for attr in ['chunks', 'elements', 'blocks', 'regions']:
                    if hasattr(result, attr):
                        chunks = getattr(result, attr)
                        logger.info(f"Found chunks via attribute: {attr}")
                        break
        
        if not chunks:
            logger.warning("No chunks found in Pix2Text result")
            
            # Try to get markdown/text output as fallback
            markdown_text = None
            if hasattr(result, 'text'):
                markdown_text = result.text
            elif hasattr(result, 'markdown'):
                markdown_text = result.markdown
            elif isinstance(result, str):
                markdown_text = result
            elif isinstance(result, dict):
                markdown_text = result.get('text') or result.get('markdown')
            
            if markdown_text:
                logger.info(f"Using Pix2Text markdown output (length: {len(markdown_text)})")
                # Parse markdown into blocks (simple heuristic)
                lines = markdown_text.split('\n')
                line_height = h // max(len(lines), 1)
                
                for i, line in enumerate(lines):
                    if line.strip():
                        y_pos = i * line_height
                        block_type = BlockType.PARAGRAPH
                        
                        # Detect equations in markdown (LaTeX between $ or $$)
                        if '$$' in line or (line.strip().startswith('$') and line.strip().endswith('$')):
                            block_type = BlockType.EQUATION_BLOCK
                        
                        blocks.append(Block(
                            bbox=BoundingBox(0, y_pos, w, min(y_pos + line_height, h)),
                            block_type=block_type,
                            confidence=0.7,
                            block_id=f"p2t_md_{i}",
                            text=line.strip() if block_type == BlockType.PARAGRAPH else None,
                            metadata={"source": "markdown", "content": line.strip()}
                        ))
            else:
                # Last resort: create a single block covering the whole page
                logger.warning("Pix2Text returned no usable output - creating fallback block")
                blocks.append(Block(
                    bbox=BoundingBox(0, 0, w, h),
                    block_type=BlockType.PARAGRAPH,
                    confidence=0.3,
                    block_id="p2t_fallback",
                    metadata={"note": "Pix2Text returned no chunks or text"}
                ))
        else:
            for i, chunk in enumerate(chunks):
                try:
                    # Get chunk type - Pix2Text uses 'text', 'formula', 'table', etc.
                    chunk_type = None
                    if hasattr(chunk, 'type'):
                        chunk_type = chunk.type
                    elif isinstance(chunk, dict):
                        chunk_type = chunk.get('type', chunk.get('category', 'text'))
                    else:
                        # Try to infer from content
                        chunk_type = 'text'
                    
                    # Map Pix2Text types to our block types
                    # Pix2Text uses: 'text', 'formula', 'table', 'figure', etc.
                    type_map = {
                        'text': BlockType.PARAGRAPH,
                        'formula': BlockType.EQUATION_BLOCK,
                        'equation': BlockType.EQUATION_BLOCK,
                        'table': BlockType.TABLE,
                        'figure': BlockType.FIGURE,
                        'title': BlockType.TITLE,
                    }
                    block_type = type_map.get(chunk_type.lower(), BlockType.PARAGRAPH)
                    
                    # Get bbox - Pix2Text provides bbox as [x1, y1, x2, y2] or similar
                    bbox_data = None
                    if hasattr(chunk, 'bbox'):
                        bbox_data = chunk.bbox
                    elif hasattr(chunk, 'box'):
                        bbox_data = chunk.box
                    elif isinstance(chunk, dict):
                        bbox_data = chunk.get('bbox') or chunk.get('box')
                    
                    if bbox_data:
                        if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                            x1, y1, x2, y2 = bbox_data[:4]
                            bbox = BoundingBox(int(x1), int(y1), int(x2), int(y2))
                        elif hasattr(bbox_data, '__iter__'):
                            # Try to unpack if it's a nested structure
                            coords = list(bbox_data)
                            if len(coords) >= 4:
                                bbox = BoundingBox(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                            else:
                                bbox = BoundingBox(0, i * 50, w, (i + 1) * 50)
                        else:
                            bbox = BoundingBox(0, i * 50, w, (i + 1) * 50)
                    else:
                        # Estimate bbox based on position
                        y_pos = (i * h) // max(len(chunks), 1)
                        bbox = BoundingBox(0, y_pos, w, min(y_pos + 50, h))
                    
                    # Get content/text
                    content = ""
                    if hasattr(chunk, 'text'):
                        content = chunk.text or ""
                    elif hasattr(chunk, 'content'):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict):
                        content = chunk.get('text', chunk.get('content', ''))
                    else:
                        content = str(chunk) if chunk else ""
                    
                    # Get confidence/score
                    confidence = 0.8
                    if hasattr(chunk, 'score'):
                        confidence = chunk.score
                    elif hasattr(chunk, 'confidence'):
                        confidence = chunk.confidence
                    elif isinstance(chunk, dict):
                        confidence = chunk.get('score', chunk.get('confidence', 0.8))
                    
                    block = Block(
                        bbox=bbox,
                        block_type=block_type,
                        confidence=float(confidence),
                        block_id=f"p2t_{i}",
                        text=content if block_type == BlockType.PARAGRAPH and content else None,
                        metadata={
                            "pix2text_type": chunk_type,
                            "content": content
                        }
                    )
                    
                    # Special handling for equations - Pix2Text may have LaTeX
                    if block_type in [BlockType.EQUATION_BLOCK, BlockType.EQUATION_INLINE]:
                        latex = None
                        if hasattr(chunk, 'latex'):
                            latex = chunk.latex
                        elif hasattr(chunk, 'formula'):
                            latex = chunk.formula
                        elif isinstance(chunk, dict):
                            latex = chunk.get('latex') or chunk.get('formula')
                        
                        if latex:
                            block.metadata['latex'] = latex
                            block.text = None  # Don't store as text
                    
                    # Special handling for tables
                    if block_type == BlockType.TABLE:
                        table_data = None
                        if hasattr(chunk, 'table'):
                            table_data = chunk.table
                        elif isinstance(chunk, dict):
                            table_data = chunk.get('table')
                        
                        if table_data:
                            block.metadata['table_data'] = table_data
                    
                    blocks.append(block)
                    logger.debug(f"Created block {i}: {block_type.value} at {bbox}")
                    
                except Exception as e:
                    logger.warning(f"Error parsing Pix2Text chunk {i}: {e}", exc_info=True)
                    continue
        
        # Create debug image if requested
        debug_image = None
        if debug:
            debug_image = self._draw_debug(image, blocks)
        
        return LayoutResult(
            blocks=blocks,
            page_width=w,
            page_height=h,
            confidence=np.mean([b.confidence for b in blocks]) if blocks else 0.0,
            debug_image=debug_image,
            method_used="pix2text"
        )
    
    def _draw_debug(self, image: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """Draw debug visualization."""
        import cv2
        
        if len(image.shape) == 2:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = image.copy()
        
        colors = {
            BlockType.TITLE: (0, 255, 0),
            BlockType.PARAGRAPH: (255, 0, 0),
            BlockType.TABLE: (0, 0, 255),
            BlockType.FIGURE: (255, 255, 0),
            BlockType.EQUATION_BLOCK: (255, 0, 255),
            BlockType.EQUATION_INLINE: (255, 128, 255),
        }
        
        for block in blocks:
            color = colors.get(block.block_type, (128, 128, 128))
            cv2.rectangle(
                debug_img,
                (block.bbox.x1, block.bbox.y1),
                (block.bbox.x2, block.bbox.y2),
                color,
                2
            )
            cv2.putText(
                debug_img,
                f"{block.block_type.value}:{block.confidence:.2f}",
                (block.bbox.x1, block.bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return debug_img


# ============================================================================
# LayoutParser Implementation
# ============================================================================

class LayoutParserDetector:
    """Layout detection using LayoutParser with Detectron2."""
    
    LABEL_MAP = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.TITLE,
        "list": BlockType.LIST,
        "table": BlockType.TABLE,
        "figure": BlockType.FIGURE,
    }
    
    def __init__(
        self,
        model_type: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        import layoutparser as lp
        
        device = "cuda" if use_gpu else "cpu"
        
        # Initialize the model
        self.model = lp.Detectron2LayoutModel(
            model_type,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", confidence_threshold],
            label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        )
        self.confidence_threshold = confidence_threshold
    
    def detect(self, image: np.ndarray, debug: bool = False) -> LayoutResult:
        """Detect layout using LayoutParser."""
        import layoutparser as lp
        import cv2
        
        # Ensure RGB format for layoutparser
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        layout = self.model.detect(rgb_image)
        
        # Convert to our Block format
        blocks = []
        for i, block in enumerate(layout):
            bbox = BoundingBox(
                int(block.block.x_1),
                int(block.block.y_1),
                int(block.block.x_2),
                int(block.block.y_2)
            )
            
            block_type = self.LABEL_MAP.get(block.type, BlockType.UNKNOWN)
            
            blocks.append(Block(
                bbox=bbox,
                block_type=block_type,
                confidence=float(block.score),
                block_id=f"lp_{i}"
            ))
        
        # Create debug image if requested
        debug_image = None
        if debug:
            debug_image = lp.draw_box(rgb_image, layout)
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        
        h, w = image.shape[:2]
        return LayoutResult(
            blocks=blocks,
            page_width=w,
            page_height=h,
            confidence=np.mean([b.confidence for b in blocks]) if blocks else 0.0,
            debug_image=debug_image
        )


# ============================================================================
# PaddleOCR Layout Implementation
# ============================================================================

class PaddleLayoutDetector:
    """Layout detection using PaddleOCR."""
    
    LABEL_MAP = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.TITLE,
        "figure": BlockType.FIGURE,
        "figure_caption": BlockType.CAPTION,
        "table": BlockType.TABLE,
        "table_caption": BlockType.CAPTION,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "reference": BlockType.REFERENCES,
        "equation": BlockType.EQUATION_BLOCK,
    }
    
    def __init__(self, use_gpu: bool = False):
        try:
            from paddleocr import PPStructure
            import logging
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            
            # Try new API first, fall back to old API
            try:
                self.engine = PPStructure(
                    layout=True,
                    table=False,
                    ocr=False,
                    use_gpu=use_gpu
                )
            except TypeError:
                # Older version with show_log
                self.engine = PPStructure(
                    layout=True,
                    table=False,
                    ocr=False,
                    show_log=False,
                    use_gpu=use_gpu
                )
        except ImportError:
            raise ImportError(
                "PaddleOCR PPStructure not available. "
                "Try: pip install 'paddleocr>=2.6'"
            )
    
    def detect(self, image: np.ndarray, debug: bool = False) -> LayoutResult:
        """Detect layout using PaddleOCR."""
        import cv2
        
        # Run detection
        result = self.engine(image)
        
        # Convert to our Block format
        blocks = []
        for i, item in enumerate(result):
            if 'type' in item and 'bbox' in item:
                x1, y1, x2, y2 = item['bbox']
                bbox = BoundingBox(int(x1), int(y1), int(x2), int(y2))
                
                block_type = self.LABEL_MAP.get(item['type'], BlockType.UNKNOWN)
                confidence = item.get('score', 0.5)
                
                blocks.append(Block(
                    bbox=bbox,
                    block_type=block_type,
                    confidence=float(confidence),
                    block_id=f"pp_{i}"
                ))
        
        # Create debug image if requested
        debug_image = None
        if debug:
            debug_image = self._draw_debug(image, blocks)
        
        h, w = image.shape[:2]
        return LayoutResult(
            blocks=blocks,
            page_width=w,
            page_height=h,
            confidence=np.mean([b.confidence for b in blocks]) if blocks else 0.0,
            debug_image=debug_image
        )
    
    def _draw_debug(self, image: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """Draw debug visualization."""
        import cv2
        
        if len(image.shape) == 2:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = image.copy()
        
        colors = {
            BlockType.TITLE: (0, 255, 0),
            BlockType.PARAGRAPH: (255, 0, 0),
            BlockType.TABLE: (0, 0, 255),
            BlockType.FIGURE: (255, 255, 0),
            BlockType.EQUATION_BLOCK: (255, 0, 255),
        }
        
        for block in blocks:
            color = colors.get(block.block_type, (128, 128, 128))
            cv2.rectangle(
                debug_img,
                (block.bbox.x1, block.bbox.y1),
                (block.bbox.x2, block.bbox.y2),
                color,
                2
            )
            cv2.putText(
                debug_img,
                f"{block.block_type.value}:{block.confidence:.2f}",
                (block.bbox.x1, block.bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return debug_img


# ============================================================================
# Classical CV Layout Detection (Fallback)
# ============================================================================

class ClassicalLayoutDetector:
    """
    Classical computer vision based layout detection.
    
    Uses connected components and heuristics to detect blocks.
    This is a fallback when deep learning models are not available.
    """
    
    def __init__(self, min_block_area: int = 100):
        self.min_block_area = min_block_area
    
    def detect(self, image: np.ndarray, debug: bool = False) -> LayoutResult:
        """Detect layout using classical CV methods."""
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text into blocks
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        
        dilated = cv2.dilate(binary, kernel_horizontal, iterations=2)
        dilated = cv2.dilate(dilated, kernel_vertical, iterations=2)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
        
        blocks = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, bw, bh, area = stats[i]
            
            if area < self.min_block_area:
                continue
            
            # Skip very thin components
            if bw < 20 or bh < 10:
                continue
            
            bbox = BoundingBox.from_xywh(x, y, bw, bh)
            
            # Classify based on heuristics (pass image for equation detection)
            block_type = self._classify_block(bbox, w, h, gray)
            
            blocks.append(Block(
                bbox=bbox,
                block_type=block_type,
                confidence=0.7,  # Lower confidence for heuristic detection
                block_id=f"cv_{i}"
            ))
        
        # Merge overlapping blocks
        blocks = self._merge_overlapping_blocks(blocks)
        
        # Create debug image
        debug_image = None
        if debug:
            from .images import draw_debug_image
            boxes = [b.bbox.to_xywh() for b in blocks]
            labels_list = [b.block_type.value for b in blocks]
            debug_image = draw_debug_image(image, boxes, labels_list)
        
        return LayoutResult(
            blocks=blocks,
            page_width=w,
            page_height=h,
            confidence=0.7,
            debug_image=debug_image
        )
    
    def _classify_block(
        self,
        bbox: BoundingBox,
        page_width: int,
        page_height: int,
        image: np.ndarray = None
    ) -> BlockType:
        """Classify a block based on position and size heuristics."""
        # Relative dimensions
        rel_width = bbox.width / page_width
        rel_height = bbox.height / page_height
        rel_y = bbox.y1 / page_height
        
        # Calculate margins (for centering detection)
        margin_left = bbox.x1 / page_width
        margin_right = (page_width - bbox.x2) / page_width
        is_centered = abs(margin_left - margin_right) < 0.15 and margin_left > 0.1
        
        # Title: wide, near top, not too tall
        if rel_y < 0.15 and rel_width > 0.4 and rel_height < 0.08:
            return BlockType.TITLE
        
        # Header: very near top, across page
        if rel_y < 0.05:
            return BlockType.HEADER
        
        # Footer: very near bottom
        if bbox.y2 > page_height * 0.95:
            return BlockType.FOOTER
        
        # EQUATION detection - centered, short height, moderate width
        # Equations are typically centered, single-line, and not full width
        if is_centered and rel_height < 0.06 and 0.15 < rel_width < 0.7:
            return BlockType.EQUATION_BLOCK
        
        # Also detect equations by checking if block has equation-like characteristics
        if image is not None:
            if self._looks_like_equation(bbox, image, page_width, page_height):
                return BlockType.EQUATION_BLOCK
        
        # Figure: roughly square, significant size
        aspect_ratio = bbox.width / max(bbox.height, 1)
        if 0.5 < aspect_ratio < 2.0 and bbox.area > (page_width * page_height) * 0.05:
            return BlockType.FIGURE
        
        # Default to paragraph
        return BlockType.PARAGRAPH
    
    def _looks_like_equation(
        self,
        bbox: BoundingBox,
        image: np.ndarray,
        page_width: int,
        page_height: int
    ) -> bool:
        """Check if a block looks like an equation based on image analysis."""
        import cv2
        
        # Extract block region
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        block_img = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if block_img.size == 0:
            return False
        
        h, w = block_img.shape
        
        # Equations typically have:
        # 1. High white space ratio (60-95%)
        _, binary = cv2.threshold(block_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = np.sum(binary > 200) / binary.size
        
        # 2. Are relatively short (not multi-line paragraphs)
        is_short = h < page_height * 0.08
        
        # 3. Are reasonably centered
        margin_left = bbox.x1 / page_width
        margin_right = (page_width - bbox.x2) / page_width
        is_centered = abs(margin_left - margin_right) < 0.2
        
        # 4. Have sparse content (lots of spacing typical in math)
        if 0.55 < white_ratio < 0.95 and is_short and is_centered:
            return True
        
        return False
    
    def _merge_overlapping_blocks(
        self,
        blocks: List[Block],
        iou_threshold: float = 0.5
    ) -> List[Block]:
        """Merge blocks with high overlap."""
        if len(blocks) <= 1:
            return blocks
        
        merged = []
        used = [False] * len(blocks)
        
        for i, block_i in enumerate(blocks):
            if used[i]:
                continue
            
            current_bbox = block_i.bbox
            
            for j, block_j in enumerate(blocks[i+1:], i+1):
                if used[j]:
                    continue
                
                if current_bbox.iou(block_j.bbox) > iou_threshold:
                    # Merge: expand bbox to include both
                    current_bbox = BoundingBox(
                        min(current_bbox.x1, block_j.bbox.x1),
                        min(current_bbox.y1, block_j.bbox.y1),
                        max(current_bbox.x2, block_j.bbox.x2),
                        max(current_bbox.y2, block_j.bbox.y2)
                    )
                    used[j] = True
            
            block_i.bbox = current_bbox
            merged.append(block_i)
            used[i] = True
        
        return merged


# ============================================================================
# Equation Detection
# ============================================================================

def detect_equations(
    image: np.ndarray,
    text_blocks: List[Block]
) -> List[Block]:
    """
    Detect equation regions within the image.
    
    Uses heuristics to find:
    - Block equations (standalone, centered)
    - Inline equations (within text lines)
    
    Args:
        image: Input image
        text_blocks: Already detected text blocks
        
    Returns:
        List of equation blocks
    """
    import cv2
    
    equations = []
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Find regions with mathematical symbols using simple pattern matching
    # This is a heuristic approach - deep learning models work better
    
    # Look for centered, short blocks (likely display equations)
    for block in text_blocks:
        if block.block_type == BlockType.PARAGRAPH:
            # Check if block is centered
            margin_left = block.bbox.x1
            margin_right = w - block.bbox.x2
            is_centered = abs(margin_left - margin_right) < w * 0.1
            
            # Check if block is short (single line equation)
            is_short = block.bbox.height < h * 0.05
            
            # Check if it has significant whitespace (typical of equations)
            block_img = gray[block.bbox.y1:block.bbox.y2, block.bbox.x1:block.bbox.x2]
            white_ratio = np.sum(block_img > 200) / block_img.size
            
            if is_centered and is_short and white_ratio > 0.7:
                eq_block = Block(
                    bbox=block.bbox,
                    block_type=BlockType.EQUATION_BLOCK,
                    confidence=0.6,
                    block_id=f"eq_{len(equations)}"
                )
                equations.append(eq_block)
    
    return equations


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
        
        # Detect layout
        detector = LayoutDetector(
            confidence_threshold=0.5,
            use_gpu=False,
            fallback_to_classical=True
        )
        
        result = detector.detect(image, debug=True)
        
        print(f"Detection method: {result.method_used}")
        print(f"Found {len(result.blocks)} blocks")
        print(f"Detected {result.num_columns} column(s)")
        
        for block in result.blocks:
            print(f"  [{block.reading_order}] {block.block_type.value}: "
                  f"bbox={block.bbox.to_tuple()}, conf={block.confidence:.2f}")
        
        # Save debug image
        if result.debug_image is not None:
            cv2.imwrite("layout_debug.png", result.debug_image)
            print("Saved debug image to layout_debug.png")
    else:
        print("Usage: python layout.py <image_path>")

