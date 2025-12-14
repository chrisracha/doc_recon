"""
Configuration and constants for the document reconstruction pipeline.

This module provides:
- Global configuration settings
- Model paths and URLs
- API configuration for optional cloud services
- Processing parameters
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("doc_recon")


# ============================================================================
# Directory Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent
MODELS_DIR = SRC_DIR / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Processing Configuration
# ============================================================================

@dataclass
class ImageConfig:
    """Image preprocessing configuration."""
    target_dpi: int = 300
    max_dpi: int = 400
    denoise_strength: int = 10
    binarize_threshold: int = 0  # 0 = auto (Otsu's method)
    deskew_max_angle: float = 15.0
    contrast_clip_limit: float = 2.0
    contrast_grid_size: int = 8
    min_block_area: int = 100  # Minimum area for detected blocks
    

@dataclass
class LayoutConfig:
    """Layout detection configuration."""
    model_type: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    confidence_threshold: float = 0.5
    use_gpu: bool = False
    fallback_to_classical: bool = True
    # Block type mappings from model labels
    label_map: Dict[str, str] = field(default_factory=lambda: {
        "text": "paragraph",
        "title": "title",
        "list": "paragraph",
        "table": "table",
        "figure": "figure",
    })


@dataclass
class OCRConfig:
    """OCR configuration."""
    # Primary OCR engine
    primary_engine: str = "tesseract"  # tesseract, paddleocr, easyocr
    # Secondary OCR for low-confidence regions
    secondary_engine: Optional[str] = "easyocr"
    # Tesseract configuration
    tesseract_lang: str = "eng"
    tesseract_config: str = "--oem 3 --psm 6"
    # Confidence thresholds
    high_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.65
    # Post-processing
    fix_hyphenation: bool = True
    merge_lines: bool = True


@dataclass
class MathOCRConfig:
    """Math/Equation OCR configuration."""
    # Primary math OCR engine
    engine: str = "pix2tex"  # pix2tex, mathpix
    # Mathpix API (optional)
    mathpix_app_id: Optional[str] = None
    mathpix_app_key: Optional[str] = None
    # pix2tex model settings
    pix2tex_model: str = "default"
    # Confidence threshold for equations
    confidence_threshold: float = 0.65
    # Maximum equation dimensions (to filter noise)
    max_equation_width: int = 2000
    max_equation_height: int = 500
    min_equation_width: int = 20
    min_equation_height: int = 10


@dataclass
class TableConfig:
    """Table extraction configuration."""
    # Detection method: camelot (PDF), image_based, hybrid
    detection_method: str = "hybrid"
    # Camelot settings (for PDF)
    camelot_flavor: str = "lattice"  # lattice or stream
    # Image-based settings
    min_rows: int = 2
    min_cols: int = 2
    line_scale: int = 40
    # Output formats
    output_formats: List[str] = field(default_factory=lambda: [
        "markdown", "html", "csv", "json"
    ])


@dataclass
class ExportConfig:
    """Export configuration."""
    # Markdown settings
    markdown_math_delimiter: str = "$"  # $ for inline, $$ for block
    # DOCX settings
    docx_template: Optional[str] = None
    docx_equation_as_image: bool = True
    # LaTeX/PDF settings
    latex_document_class: str = "article"
    latex_packages: List[str] = field(default_factory=lambda: [
        "amsmath", "amssymb", "graphicx", "booktabs", "hyperref"
    ])
    pdflatex_path: str = "pdflatex"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    image: ImageConfig = field(default_factory=ImageConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    math_ocr: MathOCRConfig = field(default_factory=MathOCRConfig)
    table: TableConfig = field(default_factory=TableConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Global settings
    use_gpu: bool = False
    debug_mode: bool = False
    output_debug_images: bool = False
    max_pages: Optional[int] = None  # None = process all pages
    

# ============================================================================
# Default Configuration Instance
# ============================================================================

def get_config() -> PipelineConfig:
    """Get the default pipeline configuration with environment overrides."""
    config = PipelineConfig()
    
    # Override from environment variables
    if os.environ.get("DOC_RECON_USE_GPU", "").lower() == "true":
        config.use_gpu = True
        config.layout.use_gpu = True
    
    if os.environ.get("DOC_RECON_DEBUG", "").lower() == "true":
        config.debug_mode = True
        config.output_debug_images = True
    
    # Mathpix API credentials from environment
    config.math_ocr.mathpix_app_id = os.environ.get("MATHPIX_APP_ID")
    config.math_ocr.mathpix_app_key = os.environ.get("MATHPIX_APP_KEY")
    
    return config


# ============================================================================
# Block Types Enumeration
# ============================================================================

class BlockType:
    """Standard block type identifiers."""
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


# ============================================================================
# Model URLs and Checksums
# ============================================================================

MODEL_URLS = {
    "layoutparser_publaynet": {
        "url": "https://layout-parser.github.io/lp-hub-static/models/PubLayNet/faster_rcnn_R_50_FPN_3x/model_final.pth",
        "checksum": None,
        "description": "PubLayNet Faster R-CNN model for document layout detection"
    },
    "pix2tex": {
        "url": None,  # Automatically downloaded by pix2tex package
        "description": "pix2tex LaTeX OCR model"
    }
}


# ============================================================================
# JSON Schema Version
# ============================================================================

JSON_SCHEMA_VERSION = "1.0"


# ============================================================================
# Utility Functions
# ============================================================================

def check_gpu_available() -> bool:
    """Check if GPU is available for inference."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Get the appropriate device string for PyTorch."""
    if check_gpu_available():
        return "cuda"
    return "cpu"

