"""
Table extraction module for document reconstruction.

Provides:
- Table detection and extraction from images
- Cell segmentation
- Multiple output formats (Markdown, HTML, CSV, JSON)
- Support for Camelot (PDF) and image-based methods
"""

import logging
import csv
import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Cell:
    """A single table cell."""
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 1.0
    is_header: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header,
            "confidence": self.confidence
        }


@dataclass
class TableResult:
    """Result of table extraction."""
    cells: List[Cell]
    num_rows: int
    num_cols: int
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    method_used: str = ""
    status: str = "success"  # success, partial, failed
    
    # Pre-generated output formats
    table_markdown: str = ""
    table_html: str = ""
    table_csv: str = ""
    table_struct: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.table_struct:
            self.table_struct = self._build_struct()
        if not self.table_markdown:
            self.table_markdown = self._build_markdown()
        if not self.table_html:
            self.table_html = self._build_html()
        if not self.table_csv:
            self.table_csv = self._build_csv()
    
    def _build_struct(self) -> Dict[str, Any]:
        """Build structured representation."""
        # Create 2D grid
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text
        
        return {
            "rows": grid,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "headers": grid[0] if self.num_rows > 0 else []
        }
    
    def _build_markdown(self) -> str:
        """Build Markdown table representation."""
        if self.num_rows == 0 or self.num_cols == 0:
            return ""
        
        grid = self.table_struct.get("rows", [])
        if not grid:
            return ""
        
        lines = []
        
        # Header row
        header = "| " + " | ".join(str(c) for c in grid[0]) + " |"
        lines.append(header)
        
        # Separator
        separator = "| " + " | ".join("---" for _ in range(self.num_cols)) + " |"
        lines.append(separator)
        
        # Data rows
        for row in grid[1:]:
            line = "| " + " | ".join(str(c) for c in row) + " |"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _build_html(self) -> str:
        """Build HTML table representation."""
        if self.num_rows == 0 or self.num_cols == 0:
            return ""
        
        grid = self.table_struct.get("rows", [])
        if not grid:
            return ""
        
        lines = ['<table>']
        
        # Header
        lines.append('  <thead>')
        lines.append('    <tr>')
        for cell in grid[0]:
            lines.append(f'      <th>{self._escape_html(str(cell))}</th>')
        lines.append('    </tr>')
        lines.append('  </thead>')
        
        # Body
        lines.append('  <tbody>')
        for row in grid[1:]:
            lines.append('    <tr>')
            for cell in row:
                lines.append(f'      <td>{self._escape_html(str(cell))}</td>')
            lines.append('    </tr>')
        lines.append('  </tbody>')
        
        lines.append('</table>')
        
        return "\n".join(lines)
    
    def _build_csv(self) -> str:
        """Build CSV representation."""
        if self.num_rows == 0 or self.num_cols == 0:
            return ""
        
        grid = self.table_struct.get("rows", [])
        if not grid:
            return ""
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        for row in grid:
            writer.writerow([str(c) for c in row])
        
        return output.getvalue()
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "confidence": self.confidence,
            "status": self.status,
            "method": self.method_used,
            "cells": [c.to_dict() for c in self.cells],
            "markdown": self.table_markdown,
            "html": self.table_html,
            "csv": self.table_csv,
            "struct": self.table_struct
        }


# ============================================================================
# Table Extractor Main Class
# ============================================================================

class TableExtractor:
    """
    Main table extraction interface.
    
    Supports:
    - Camelot (for PDF files with vector tables)
    - Image-based detection using OpenCV
    - Deep learning models (when available)
    """
    
    def __init__(
        self,
        method: str = "hybrid",
        use_gpu: bool = False,
        min_rows: int = 2,
        min_cols: int = 2
    ):
        self.method = method
        self.use_gpu = use_gpu
        self.min_rows = min_rows
        self.min_cols = min_cols
        
        self._extractors = {}
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize available extractors."""
        # Always have OpenCV fallback
        self._extractors["opencv"] = OpenCVTableExtractor(
            min_rows=self.min_rows,
            min_cols=self.min_cols
        )
        
        # Try to load Camelot
        try:
            self._extractors["camelot"] = CamelotTableExtractor()
            logger.info("Camelot available for PDF table extraction")
        except ImportError:
            logger.warning("Camelot not available")
        
        # Try to load deep learning extractor
        try:
            self._extractors["deep"] = DeepTableExtractor(use_gpu=self.use_gpu)
            logger.info("Deep learning table extractor available")
        except Exception as e:
            logger.warning(f"Deep learning table extractor not available: {e}")
    
    def extract(
        self,
        image: np.ndarray,
        pdf_path: Optional[Union[str, Path]] = None,
        page_num: int = 1
    ) -> List[TableResult]:
        """
        Extract tables from image or PDF.
        
        Args:
            image: Page image (required for image-based extraction)
            pdf_path: Optional PDF path (for Camelot extraction)
            page_num: Page number in PDF (1-indexed)
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        # Try Camelot for PDFs first
        if pdf_path and "camelot" in self._extractors:
            try:
                camelot_results = self._extractors["camelot"].extract(
                    pdf_path, page_num
                )
                if camelot_results:
                    results.extend(camelot_results)
                    logger.info(f"Camelot found {len(camelot_results)} tables")
            except Exception as e:
                logger.warning(f"Camelot extraction failed: {e}")
        
        # If no results from Camelot, try image-based methods
        if not results:
            if "deep" in self._extractors and self.method in ["deep", "hybrid"]:
                try:
                    deep_results = self._extractors["deep"].extract(image)
                    if deep_results:
                        results.extend(deep_results)
                        logger.info(f"Deep learning found {len(deep_results)} tables")
                except Exception as e:
                    logger.warning(f"Deep learning extraction failed: {e}")
            
            # Fall back to OpenCV
            if not results or self.method in ["opencv", "hybrid"]:
                try:
                    cv_results = self._extractors["opencv"].extract(image)
                    if cv_results:
                        # Don't duplicate if deep learning already found tables
                        if not results:
                            results.extend(cv_results)
                        logger.info(f"OpenCV found {len(cv_results)} tables")
                except Exception as e:
                    logger.warning(f"OpenCV extraction failed: {e}")
        
        return results
    
    def extract_from_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[TableResult]:
        """
        Extract a single table from a specific region.
        
        Args:
            image: Full page image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            TableResult or None if extraction fails
        """
        x1, y1, x2, y2 = bbox
        table_image = image[y1:y2, x1:x2]
        
        if table_image.size == 0:
            return None
        
        results = self.extract(table_image)
        
        if results:
            result = results[0]
            # Adjust bbox to page coordinates
            result.bbox = bbox
            return result
        
        return None


# ============================================================================
# Camelot Table Extractor (PDF)
# ============================================================================

class CamelotTableExtractor:
    """Table extraction from PDFs using Camelot."""
    
    def __init__(self, flavor: str = "lattice"):
        try:
            import camelot
            self.camelot = camelot
        except ImportError:
            raise ImportError(
                "Camelot not available. Install with: pip install camelot-py[cv]"
            )
        
        self.flavor = flavor
    
    def extract(
        self,
        pdf_path: Union[str, Path],
        page_num: int = 1
    ) -> List[TableResult]:
        """Extract tables from PDF page."""
        try:
            tables = self.camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num),
                flavor=self.flavor
            )
        except Exception as e:
            logger.error(f"Camelot error: {e}")
            return []
        
        results = []
        for table in tables:
            df = table.df
            
            cells = []
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_text = str(df.iloc[row_idx, col_idx])
                    cells.append(Cell(
                        text=cell_text,
                        row=row_idx,
                        col=col_idx,
                        is_header=(row_idx == 0)
                    ))
            
            result = TableResult(
                cells=cells,
                num_rows=len(df),
                num_cols=len(df.columns),
                confidence=table.accuracy / 100.0,
                method_used="camelot"
            )
            results.append(result)
        
        return results


# ============================================================================
# OpenCV Table Extractor (Image-based)
# ============================================================================

class OpenCVTableExtractor:
    """Table extraction from images using OpenCV."""
    
    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        line_scale: int = 40
    ):
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.line_scale = line_scale
    
    def extract(self, image: np.ndarray) -> List[TableResult]:
        """Extract tables from image using line detection."""
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Minimum size requirements for tables
        if h < 80 or w < 150:
            return []  # Too small to be a table
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal and vertical lines with stricter parameters
        # Use longer kernels to ensure we detect actual table lines, not text characters
        h_kernel_size = max(w // 10, 50)  # At least 50 pixels or 10% of width
        v_kernel_size = max(h // 10, 30)  # At least 30 pixels or 10% of height
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Count line pixels to verify we have actual table structure
        h_pixels = cv2.countNonZero(horizontal_lines)
        v_pixels = cv2.countNonZero(vertical_lines)
        
        # Need minimum line coverage for a real table
        h_coverage = h_pixels / (w * h)
        v_coverage = v_pixels / (w * h)
        
        if h_coverage < 0.005 or v_coverage < 0.005:
            return []  # Not enough line structure for a table
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours (cells)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours - stricter filtering
        cell_contours = []
        min_cell_area = max(500, (w * h) * 0.005)  # At least 500 pixels or 0.5% of image
        max_cell_area = (w * h) * 0.5  # No more than 50% of image
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            # Filter by size - need reasonable cell size
            if min_cell_area < area < max_cell_area:
                # Cell aspect ratio check - cells shouldn't be too extreme
                aspect = cw / max(ch, 1)
                if 0.2 < aspect < 10:
                    cell_contours.append((x, y, cw, ch))
        
        if len(cell_contours) < self.min_rows * self.min_cols:
            return []
        
        # Try to organize into grid
        cells, num_rows, num_cols = self._organize_into_grid(cell_contours, image)
        
        if num_rows < self.min_rows or num_cols < self.min_cols:
            return []
        
        result = TableResult(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            confidence=0.7,
            bbox=(0, 0, w, h),
            method_used="opencv"
        )
        
        return [result]
    
    def _organize_into_grid(
        self,
        contours: List[Tuple[int, int, int, int]],
        image: np.ndarray
    ) -> Tuple[List[Cell], int, int]:
        """Organize contours into a grid structure."""
        import cv2
        
        if not contours:
            return [], 0, 0
        
        # Sort by y, then x
        sorted_contours = sorted(contours, key=lambda c: (c[1], c[0]))
        
        # Group by rows (similar y values)
        rows = []
        current_row = [sorted_contours[0]]
        threshold = image.shape[0] * 0.02  # 2% of image height
        
        for cnt in sorted_contours[1:]:
            if abs(cnt[1] - current_row[0][1]) < threshold:
                current_row.append(cnt)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cnt]
        rows.append(sorted(current_row, key=lambda c: c[0]))
        
        # Extract text from cells
        cells = []
        num_cols = max(len(row) for row in rows) if rows else 0
        
        for row_idx, row in enumerate(rows):
            for col_idx, (x, y, cw, ch) in enumerate(row):
                # Extract cell image
                cell_img = image[y:y+ch, x:x+cw]
                
                # OCR the cell
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(cell_img, config='--psm 6').strip()
                except:
                    text = ""
                
                cells.append(Cell(
                    text=text,
                    row=row_idx,
                    col=col_idx,
                    bbox=(x, y, x+cw, y+ch),
                    is_header=(row_idx == 0)
                ))
        
        return cells, len(rows), num_cols


# ============================================================================
# Deep Learning Table Extractor
# ============================================================================

class DeepTableExtractor:
    """Table extraction using deep learning models."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._model = None
        
        # Try to load table transformer or similar model
        try:
            from transformers import TableTransformerForObjectDetection, DetrImageProcessor
            import torch
            
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self._model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            
            if use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._model.eval()
            logger.info("Loaded Table Transformer model")
            
        except ImportError:
            raise ImportError(
                "Table Transformer not available. Install with: "
                "pip install transformers torch"
            )
    
    def extract(self, image: np.ndarray) -> List[TableResult]:
        """Extract tables using deep learning model."""
        import torch
        from PIL import Image
        import cv2
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        if self.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=0.7,
            target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(
            results["scores"], 
            results["labels"], 
            results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Extract table region
            table_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Use OpenCV to extract cells
            cv_extractor = OpenCVTableExtractor()
            cell_results = cv_extractor.extract(table_img)
            
            if cell_results:
                result = cell_results[0]
                result.bbox = bbox
                result.confidence = float(score)
                result.method_used = "deep_learning"
                tables.append(result)
        
        return tables


# ============================================================================
# Utility Functions
# ============================================================================

def detect_table_regions(
    image: np.ndarray,
    min_table_area: float = 0.01
) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions that likely contain tables.
    
    Args:
        image: Input image
        min_table_area: Minimum table area as fraction of page
        
    Returns:
        List of bounding boxes (x1, y1, x2, y2)
    """
    import cv2
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    min_area = (h * w) * min_table_area
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        # Check if likely a table (rectangular, sufficient size)
        if area > min_area:
            aspect = cw / ch if ch > 0 else 0
            if 0.2 < aspect < 5:  # Reasonable aspect ratio
                table_regions.append((x, y, x + cw, y + ch))
    
    return table_regions


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import cv2
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        # Check if PDF or image
        if input_path.lower().endswith('.pdf'):
            print("PDF table extraction")
            extractor = TableExtractor(method="hybrid")
            
            # Load first page as image for fallback
            from .io import load_pdf
            images = load_pdf(input_path, dpi=300)
            if images:
                results = extractor.extract(images[0], pdf_path=input_path)
        else:
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                sys.exit(1)
            
            extractor = TableExtractor(method="opencv")
            results = extractor.extract(image)
        
        print(f"Found {len(results)} tables")
        for i, table in enumerate(results):
            print(f"\nTable {i+1}:")
            print(f"  Size: {table.num_rows} x {table.num_cols}")
            print(f"  Confidence: {table.confidence:.2f}")
            print(f"  Method: {table.method_used}")
            print(f"\nMarkdown:\n{table.table_markdown}")
    else:
        print("Usage: python tables.py <image_or_pdf_path>")


