"""
Document assembler module for document reconstruction.

Provides:
- Document data model (Document, Page, Block)
- Pipeline orchestration
- JSON envelope generation
- Metrics calculation
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from .layout import Block, BlockType, BoundingBox, LayoutResult
from .ocr_text import OCRResult
from .ocr_math import EquationResult
from .tables import TableResult

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProcessedBlock:
    """A fully processed document block."""
    block_id: str
    block_type: str
    bbox: Dict[str, int]
    reading_order: int
    column_index: int
    confidence: float
    
    # Content (one of these will be populated)
    text: Optional[str] = None
    latex: Optional[str] = None
    table: Optional[Dict[str, Any]] = None
    figure_path: Optional[str] = None
    
    # Additional metadata
    alternatives: List[str] = field(default_factory=list)
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "block_id": self.block_id,
            "type": self.block_type,
            "bbox": self.bbox,
            "reading_order": self.reading_order,
            "column_index": self.column_index,
            "confidence": self.confidence,
            "status": self.status,
            "metadata": self.metadata
        }
        
        if self.text is not None:
            result["text"] = self.text
        if self.latex is not None:
            result["latex"] = self.latex
            if self.alternatives:
                result["latex_alternatives"] = self.alternatives
        if self.table is not None:
            result["table"] = self.table
        if self.figure_path is not None:
            result["figure_path"] = self.figure_path
        
        return result


@dataclass
class Page:
    """A document page with all processed blocks."""
    page_number: int
    width: int
    height: int
    blocks: List[ProcessedBlock] = field(default_factory=list)
    deskew_angle: float = 0.0
    num_columns: int = 1
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text_blocks(self) -> List[ProcessedBlock]:
        return [b for b in self.blocks if b.text is not None]
    
    @property
    def equation_blocks(self) -> List[ProcessedBlock]:
        return [b for b in self.blocks if b.latex is not None]
    
    @property
    def table_blocks(self) -> List[ProcessedBlock]:
        return [b for b in self.blocks if b.table is not None]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "num_columns": self.num_columns,
            "deskew_angle": self.deskew_angle,
            "image_path": self.image_path,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata
        }


@dataclass
class DocumentMetrics:
    """Metrics about document processing."""
    global_confidence: float = 0.0
    coverage_pct: float = 0.0
    
    # Text metrics
    text_blocks_total: int = 0
    text_blocks_high_confidence: int = 0
    text_blocks_low_confidence: int = 0
    
    # Equation metrics
    equations_total: int = 0
    equations_high_confidence: int = 0
    equations_low_confidence: int = 0
    
    # Table metrics
    tables_total: int = 0
    tables_reconstructed: int = 0
    tables_partial: int = 0
    
    # Processing stats
    pages_processed: int = 0
    processing_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_confidence": round(self.global_confidence, 3),
            "coverage_pct": round(self.coverage_pct, 3),
            "text_blocks": {
                "total": self.text_blocks_total,
                "high_confidence": self.text_blocks_high_confidence,
                "low_confidence": self.text_blocks_low_confidence
            },
            "equations": {
                "total": self.equations_total,
                "high_confidence": self.equations_high_confidence,
                "low_confidence": self.equations_low_confidence
            },
            "tables": {
                "total": self.tables_total,
                "reconstructed": self.tables_reconstructed,
                "partial": self.tables_partial
            },
            "pages_processed": self.pages_processed,
            "processing_time_seconds": round(self.processing_time_seconds, 2)
        }


@dataclass
class Document:
    """Complete reconstructed document."""
    task_id: str
    source_file: str
    pages: List[Page] = field(default_factory=list)
    metrics: Optional[DocumentMetrics] = None
    
    # Generated content
    markdown: str = ""
    docx_manifest: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    created_at: str = ""
    schema_version: str = "1.0"
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "schema_version": self.schema_version,
            "source_file": self.source_file,
            "created_at": self.created_at,
            "metadata": {
                "title": self.title or "UNREADABLE",
                "authors": self.authors or "UNREADABLE",
                "abstract": self.abstract
            },
            "pages": [p.to_dict() for p in self.pages],
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "markdown": self.markdown,
            "docx_manifest": self.docx_manifest
        }


# ============================================================================
# Document Assembler
# ============================================================================

class DocumentAssembler:
    """
    Orchestrates the document reconstruction pipeline.
    
    Coordinates:
    - Image preprocessing
    - Layout detection
    - Text OCR
    - Math OCR
    - Table extraction
    - Document assembly
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        debug_mode: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        ocr_engine: str = "tesseract",
        math_engine: str = "pix2tex",
        layout_method: str = "default"
    ):
        self.use_gpu = use_gpu
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir) if output_dir else None
        self.ocr_engine = ocr_engine
        self.math_engine = math_engine
        self.layout_method = layout_method
        
        # Initialize components lazily
        self._preprocessor = None
        self._layout_detector = None
        self._text_ocr = None
        self._math_ocr = None
        self._table_extractor = None
    
    @property
    def preprocessor(self):
        if self._preprocessor is None:
            from .images import preprocess_image
            self._preprocessor = preprocess_image
        return self._preprocessor
    
    @property
    def layout_detector(self):
        if self._layout_detector is None:
            from .layout import LayoutDetector
            self._layout_detector = LayoutDetector(
                use_gpu=self.use_gpu,
                fallback_to_classical=True,
                layout_method=self.layout_method
            )
        return self._layout_detector
    
    @property
    def text_ocr(self):
        if self._text_ocr is None:
            from .ocr_text import TextOCR
            # Use configured engine, with fallback to secondary if available
            secondary = None
            if self.ocr_engine == "tesseract":
                try:
                    import easyocr
                    secondary = "easyocr"
                except ImportError:
                    pass
            self._text_ocr = TextOCR(
                primary_engine=self.ocr_engine,
                secondary_engine=secondary,
                use_gpu=self.use_gpu
            )
        return self._text_ocr
    
    @property
    def math_ocr(self):
        if self._math_ocr is None:
            from .ocr_math import MathOCR
            self._math_ocr = MathOCR(
                engine=self.math_engine,
                use_gpu=self.use_gpu
            )
        return self._math_ocr
    
    @property
    def table_extractor(self):
        if self._table_extractor is None:
            from .tables import TableExtractor
            self._table_extractor = TableExtractor(
                method="hybrid",
                use_gpu=self.use_gpu
            )
        return self._table_extractor
    
    def process_page(
        self,
        image: np.ndarray,
        page_number: int = 1,
        pdf_path: Optional[str] = None
    ) -> Page:
        """
        Process a single page image.
        
        Args:
            image: Page image (BGR format)
            page_number: Page number (1-indexed)
            pdf_path: Optional PDF path for better table extraction
            
        Returns:
            Page object with all processed blocks
        """
        import time
        start_time = time.time()
        
        h, w = image.shape[:2]
        logger.info(f"Processing page {page_number} ({w}x{h})")
        
        # 1. Preprocess image
        preprocess_result = self.preprocessor(
            image,
            deskew_enabled=True,
            denoise_enabled=True,
            enhance_contrast_enabled=True
        )
        processed_image = preprocess_result.image
        
        # 2. Detect layout
        layout_result = self.layout_detector.detect(
            processed_image,
            debug=self.debug_mode
        )
        logger.info(f"Found {len(layout_result.blocks)} blocks")
        
        # 2.5. Try to detect equations from blocks that might be equations
        layout_result.blocks = self._detect_potential_equations(
            layout_result.blocks, 
            processed_image
        )
        
        # 2.6. Try to detect tables that weren't classified
        layout_result.blocks = self._detect_potential_tables(
            layout_result.blocks,
            processed_image
        )
        
        # 3. Process each block
        processed_blocks = []
        for block in layout_result.blocks:
            processed_block = self._process_block(
                block,
                processed_image,
                pdf_path
            )
            processed_blocks.append(processed_block)
        
        # 4. Create page object
        page = Page(
            page_number=page_number,
            width=w,
            height=h,
            blocks=processed_blocks,
            deskew_angle=preprocess_result.deskew_angle,
            num_columns=layout_result.num_columns
        )
        
        # Save debug image if enabled
        if self.debug_mode and layout_result.debug_image is not None:
            self._save_debug_image(layout_result.debug_image, page_number)
        
        elapsed = time.time() - start_time
        logger.info(f"Page {page_number} processed in {elapsed:.2f}s")
        
        return page
    
    def _detect_potential_equations(
        self,
        blocks: List[Block],
        image: np.ndarray
    ) -> List[Block]:
        """
        Post-process blocks to detect potential equations.
        
        Looks at paragraph blocks that might actually be equations based on:
        - Being centered
        - Having short height
        - Having sparse/mathematical content
        """
        import cv2
        from .layout import BlockType
        
        h, w = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        for block in blocks:
            # Only check paragraphs
            if block.block_type != BlockType.PARAGRAPH:
                continue
            
            bbox = block.bbox
            block_img = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            if block_img.size == 0:
                continue
            
            block_h, block_w = block_img.shape
            
            # Skip multi-line blocks - likely paragraphs, not equations
            # Estimate line count based on height (assuming ~20-40 pixels per line)
            estimated_lines = block_h / 25
            if estimated_lines > 2.5:
                continue  # Multi-line text blocks are not equations
            
            # Check if it looks like an equation:
            # 1. Centered on page (stricter centering)
            margin_left = bbox.x1 / w
            margin_right = (w - bbox.x2) / w
            is_centered = abs(margin_left - margin_right) < 0.15 and margin_left > 0.15
            
            # 2. Short height (single/double line only)
            is_short = block_h < h * 0.05
            
            # 3. Moderate width (not too narrow, not full width)
            rel_width = block_w / w
            has_content = 0.15 < rel_width < 0.6
            
            # 4. High white space ratio (typical of equations - math has lots of spacing)
            _, binary = cv2.threshold(block_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary > 200) / binary.size
            is_sparse = 0.65 < white_ratio < 0.95
            
            # 5. Wide aspect ratio (equations are wider than tall)
            aspect = block_w / max(block_h, 1)
            is_wide = aspect > 3.0
            
            # 6. Check for math-like symbols in text (if we can do quick OCR)
            # This is a strong indicator
            
            # Require MORE indicators for equation classification (was 3, now 4)
            indicators = sum([is_centered, is_short, is_sparse, is_wide, has_content])
            
            if indicators >= 4:
                block.block_type = BlockType.EQUATION_BLOCK
                logger.info(f"Reclassified block {block.block_id} as equation (indicators: {indicators})")
        
        return blocks
    
    def _detect_potential_tables(
        self,
        blocks: List[Block],
        image: np.ndarray
    ) -> List[Block]:
        """
        Post-process blocks to detect potential tables.
        
        Looks for blocks with grid-like structure (horizontal/vertical lines).
        """
        import cv2
        from .layout import BlockType
        
        h, w = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        for block in blocks:
            # Only check paragraphs and figures (tables can be misclassified as either)
            if block.block_type not in [BlockType.PARAGRAPH, BlockType.FIGURE, BlockType.UNKNOWN]:
                continue
            
            bbox = block.bbox
            block_img = gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            if block_img.size == 0:
                continue
            
            block_h, block_w = block_img.shape
            
            # Tables need minimum size
            if block_h < 60 or block_w < 100:
                continue
            
            # Threshold
            _, binary = cv2.threshold(block_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Detect horizontal lines
            h_kernel_size = max(block_w // 15, 15)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            
            # Detect vertical lines  
            v_kernel_size = max(block_h // 15, 15)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
            
            # Count line pixels
            h_pixels = cv2.countNonZero(horizontal_lines)
            v_pixels = cv2.countNonZero(vertical_lines)
            
            # Calculate line density
            h_density = h_pixels / (block_w * block_h) * 100
            v_density = v_pixels / (block_w * block_h) * 100
            
            # Tables need CLEAR line structure - be strict to avoid false positives
            # Both horizontal and vertical must have significant presence
            has_h_lines = h_density > 1.0  # Increased threshold
            has_v_lines = v_density > 1.0  # Increased threshold
            
            # Check for grid intersections (where h and v lines cross)
            combined = cv2.bitwise_and(horizontal_lines, vertical_lines)
            intersection_pixels = cv2.countNonZero(combined)
            
            # Require clear grid structure: strong lines AND intersections
            if has_h_lines and has_v_lines and intersection_pixels > 20:
                block.block_type = BlockType.TABLE
                logger.info(f"Reclassified block {block.block_id} as table (h_density={h_density:.2f}, v_density={v_density:.2f}, intersections={intersection_pixels})")
        
        return blocks
    
    def _process_block(
        self,
        block: Block,
        image: np.ndarray,
        pdf_path: Optional[str] = None
    ) -> ProcessedBlock:
        """Process a single block based on its type."""
        block_image = block.crop_from_image(image)
        
        processed = ProcessedBlock(
            block_id=block.block_id,
            block_type=block.block_type.value,
            bbox={
                "x1": block.bbox.x1,
                "y1": block.bbox.y1,
                "x2": block.bbox.x2,
                "y2": block.bbox.y2
            },
            reading_order=block.reading_order,
            column_index=block.column_index,
            confidence=block.confidence
        )
        
        # Process based on type
        if block.block_type in [BlockType.EQUATION_BLOCK, BlockType.EQUATION_INLINE]:
            # Math OCR
            try:
                eq_result = self.math_ocr.recognize(
                    block_image,
                    equation_type=block.block_type.value.replace("equation_", "")
                )
                processed.latex = eq_result.latex
                processed.confidence = eq_result.confidence
                processed.alternatives = eq_result.latex_alternatives
                processed.status = eq_result.status
                
                # If math OCR failed or has very low confidence, try text OCR as fallback
                if not eq_result.latex or eq_result.confidence < 0.3:
                    logger.info(f"Math OCR failed/low confidence, trying text OCR for equation")
                    try:
                        ocr_result = self.text_ocr.recognize(block_image)
                        if ocr_result.text:
                            # Store text as latex (user will see raw text)
                            processed.latex = ocr_result.text
                            processed.confidence = ocr_result.confidence * 0.5  # Lower confidence
                            processed.status = "text_fallback"
                            processed.metadata["fallback"] = "text_ocr"
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Math OCR error: {e}")
                # Try text OCR as fallback
                try:
                    ocr_result = self.text_ocr.recognize(block_image)
                    if ocr_result.text:
                        processed.latex = ocr_result.text
                        processed.confidence = ocr_result.confidence * 0.5
                        processed.status = "text_fallback"
                except:
                    processed.status = "failed"
        
        elif block.block_type == BlockType.TABLE:
            # Table extraction
            try:
                table_results = self.table_extractor.extract(
                    block_image,
                    pdf_path=pdf_path
                )
                if table_results:
                    processed.table = table_results[0].to_dict()
                    processed.confidence = table_results[0].confidence
                    processed.status = table_results[0].status
                else:
                    processed.status = "failed"
            except Exception as e:
                logger.error(f"Table extraction error: {e}")
                processed.status = "failed"
        
        elif block.block_type == BlockType.FIGURE:
            # Save figure image
            if self.output_dir:
                figure_path = self.output_dir / f"figures/{block.block_id}.png"
                figure_path.parent.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(figure_path), block_image)
                processed.figure_path = str(figure_path)
        
        else:
            # Text OCR
            try:
                ocr_result = self.text_ocr.recognize(block_image)
                processed.text = ocr_result.text
                processed.confidence = ocr_result.confidence
                processed.alternatives = ocr_result.alternatives
                # Store OCR metadata including engine fallback info
                if ocr_result.metadata:
                    processed.metadata["ocr"] = ocr_result.metadata
            except Exception as e:
                logger.error(f"Text OCR error: {e}")
                processed.text = ""
                processed.status = "failed"
        
        return processed
    
    def process_document(
        self,
        images: List[np.ndarray],
        source_file: str,
        pdf_path: Optional[str] = None
    ) -> Document:
        """
        Process a complete document.
        
        Args:
            images: List of page images
            source_file: Original source file path
            pdf_path: Optional PDF path
            
        Returns:
            Document object with all pages
        """
        import time
        start_time = time.time()
        
        doc = Document(
            task_id=str(uuid.uuid4()),
            source_file=source_file
        )
        
        # Process each page
        for i, image in enumerate(images, 1):
            page = self.process_page(image, page_number=i, pdf_path=pdf_path)
            doc.pages.append(page)
            
            # Extract document metadata from first page
            if i == 1:
                self._extract_document_metadata(doc, page)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        doc.metrics = self._calculate_metrics(doc, elapsed)
        
        # Generate markdown
        doc.markdown = self._generate_markdown(doc)
        
        # Generate DOCX manifest
        doc.docx_manifest = self._generate_docx_manifest(doc)
        
        return doc
    
    def _extract_document_metadata(self, doc: Document, page: Page):
        """Extract title, authors, abstract from first page."""
        for block in page.blocks:
            if block.block_type == "title" and block.text:
                doc.title = block.text
            elif block.block_type == "authors" and block.text:
                doc.authors = block.text
            elif block.block_type == "abstract" and block.text:
                doc.abstract = block.text
    
    def _calculate_metrics(
        self,
        doc: Document,
        processing_time: float
    ) -> DocumentMetrics:
        """Calculate document-wide metrics."""
        metrics = DocumentMetrics()
        metrics.processing_time_seconds = processing_time
        metrics.pages_processed = len(doc.pages)
        
        all_confidences = []
        total_block_area = 0
        total_page_area = 0
        
        for page in doc.pages:
            page_area = page.width * page.height
            total_page_area += page_area
            
            for block in page.blocks:
                # Block area
                block_area = (
                    (block.bbox["x2"] - block.bbox["x1"]) *
                    (block.bbox["y2"] - block.bbox["y1"])
                )
                total_block_area += block_area
                all_confidences.append(block.confidence)
                
                # Count by type
                if block.text is not None:
                    metrics.text_blocks_total += 1
                    if block.confidence >= 0.80:
                        metrics.text_blocks_high_confidence += 1
                    elif block.confidence < 0.65:
                        metrics.text_blocks_low_confidence += 1
                
                elif block.latex is not None:
                    metrics.equations_total += 1
                    if block.confidence >= 0.80:
                        metrics.equations_high_confidence += 1
                    elif block.confidence < 0.65:
                        metrics.equations_low_confidence += 1
                
                elif block.table is not None:
                    metrics.tables_total += 1
                    if block.status == "success":
                        metrics.tables_reconstructed += 1
                    elif block.status == "partial":
                        metrics.tables_partial += 1
        
        # Calculate global metrics
        if all_confidences:
            metrics.global_confidence = np.mean(all_confidences)
        
        if total_page_area > 0:
            metrics.coverage_pct = (total_block_area / total_page_area) * 100
        
        return metrics
    
    def _generate_markdown(self, doc: Document) -> str:
        """Generate Markdown representation of document."""
        lines = []
        
        # Title
        if doc.title:
            lines.append(f"# {doc.title}")
            lines.append("")
        
        # Authors
        if doc.authors:
            lines.append(f"*{doc.authors}*")
            lines.append("")
        
        # Abstract
        if doc.abstract:
            lines.append("## Abstract")
            lines.append("")
            lines.append(doc.abstract)
            lines.append("")
        
        # Content by page
        for page in doc.pages:
            if len(doc.pages) > 1:
                lines.append(f"\n---\n*Page {page.page_number}*\n")
            
            for block in sorted(page.blocks, key=lambda b: b.reading_order):
                if block.block_type == "heading":
                    lines.append(f"## {block.text}")
                    lines.append("")
                elif block.text:
                    lines.append(block.text)
                    lines.append("")
                elif block.latex:
                    lines.append(f"$$\n{block.latex}\n$$")
                    lines.append("")
                elif block.table:
                    lines.append(block.table.get("markdown", ""))
                    lines.append("")
                elif block.figure_path:
                    lines.append(f"![Figure]({block.figure_path})")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _generate_docx_manifest(self, doc: Document) -> Dict[str, Any]:
        """Generate manifest for DOCX generation."""
        elements = []
        
        # Title
        if doc.title:
            elements.append({
                "type": "heading",
                "level": 0,
                "text": doc.title
            })
        
        # Authors
        if doc.authors:
            elements.append({
                "type": "paragraph",
                "text": doc.authors,
                "style": "author"
            })
        
        # Content
        for page in doc.pages:
            for block in sorted(page.blocks, key=lambda b: b.reading_order):
                if block.block_type == "heading":
                    elements.append({
                        "type": "heading",
                        "level": 2,
                        "text": block.text
                    })
                elif block.text:
                    elements.append({
                        "type": "paragraph",
                        "text": block.text
                    })
                elif block.latex:
                    elements.append({
                        "type": "equation",
                        "latex": block.latex,
                        "display": "block" if "block" in block.block_type else "inline"
                    })
                elif block.table:
                    elements.append({
                        "type": "table",
                        "data": block.table.get("struct", {}).get("rows", [])
                    })
                elif block.figure_path:
                    elements.append({
                        "type": "image",
                        "path": block.figure_path
                    })
        
        return {
            "elements": elements,
            "metadata": {
                "title": doc.title,
                "authors": doc.authors,
                "created": doc.created_at
            }
        }
    
    def _save_debug_image(self, image: np.ndarray, page_number: int):
        """Save debug image with bounding boxes."""
        if self.output_dir:
            import cv2
            debug_path = self.output_dir / f"debug/page_{page_number:04d}_debug.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), image)
            logger.debug(f"Saved debug image: {debug_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
        
        from .io import load_pdf, load_image, detect_input_type, save_json
        
        input_type = detect_input_type(input_path)
        
        if input_type == 'pdf':
            images = load_pdf(input_path, dpi=300)
        elif input_type == 'image':
            images = [load_image(input_path)]
        else:
            print(f"Unsupported input type: {input_type}")
            sys.exit(1)
        
        # Process document
        assembler = DocumentAssembler(
            use_gpu=False,
            debug_mode=True,
            output_dir=output_dir
        )
        
        doc = assembler.process_document(
            images,
            source_file=input_path
        )
        
        # Save JSON output
        save_json(doc.to_dict(), f"{output_dir}/document.json")
        
        # Print summary
        print(f"Processed {len(doc.pages)} pages")
        print(f"Metrics: {json.dumps(doc.metrics.to_dict(), indent=2)}")
        print(f"\nMarkdown preview:\n{doc.markdown[:500]}...")
    else:
        print("Usage: python assembler.py <input_path> [output_dir]")

