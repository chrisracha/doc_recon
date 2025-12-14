"""
I/O utilities for the document reconstruction pipeline.

Handles:
- PDF loading and conversion to images
- Image loading and validation
- JSON serialization
- Directory management
"""

import json
import logging
from pathlib import Path
from typing import List, Union, Optional, Any, Dict
from dataclasses import dataclass, asdict
import tempfile
import shutil

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# PDF to Image Conversion
# ============================================================================

def load_pdf(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> List[np.ndarray]:
    """
    Convert PDF pages to images using pdf2image (poppler backend).
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (300-400 recommended for OCR)
        first_page: First page to convert (1-indexed, None = first)
        last_page: Last page to convert (1-indexed, None = last)
        output_dir: Optional directory to save intermediate images
        
    Returns:
        List of numpy arrays (BGR format) representing each page
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ImportError: If pdf2image is not installed
        RuntimeError: If poppler is not installed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
    except ImportError:
        raise ImportError(
            "pdf2image is required. Install with: pip install pdf2image\n"
            "Also ensure poppler is installed on your system."
        )
    
    try:
        logger.info(f"Converting PDF to images: {pdf_path} at {dpi} DPI")
        
        # Convert PDF to PIL images
        pil_images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='png',
            thread_count=4
        )
        
        # Convert to numpy arrays (RGB -> BGR for OpenCV compatibility)
        images = []
        for i, pil_img in enumerate(pil_images):
            img_array = np.array(pil_img)
            # Convert RGB to BGR if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1].copy()
            images.append(img_array)
            
            # Optionally save intermediate images
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(str(output_dir / f"page_{i+1:04d}.png"), img_array)
        
        logger.info(f"Converted {len(images)} pages from PDF")
        return images
        
    except (PDFPageCountError, PDFSyntaxError) as e:
        raise RuntimeError(f"Failed to parse PDF: {e}")
    except Exception as e:
        if "poppler" in str(e).lower():
            raise RuntimeError(
                "Poppler is not installed. Install with:\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows\n"
                "  macOS: brew install poppler\n"
                "  Linux: sudo apt-get install poppler-utils"
            )
        raise


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """Get the number of pages in a PDF file."""
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(pdf_path))
        return info.get('Pages', 0)
    except Exception as e:
        logger.warning(f"Could not get PDF page count: {e}")
        return 0


# ============================================================================
# Image Loading
# ============================================================================

def load_image(
    image_path: Union[str, Path],
    grayscale: bool = False
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        grayscale: If True, load as grayscale
        
    Returns:
        Numpy array representing the image (BGR format if color)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be decoded
    """
    import cv2
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(image_path), flag)
    
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    
    logger.debug(f"Loaded image: {image_path}, shape: {img.shape}")
    return img


def load_images_from_folder(
    folder_path: Union[str, Path],
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'),
    sort: bool = True
) -> List[np.ndarray]:
    """
    Load all images from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        extensions: Tuple of valid image extensions
        sort: If True, sort files alphabetically
        
    Returns:
        List of numpy arrays representing images
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    
    # Find all image files
    image_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in extensions
    ]
    
    if sort:
        image_files = sorted(image_files)
    
    logger.info(f"Found {len(image_files)} images in {folder_path}")
    
    images = []
    for img_path in image_files:
        try:
            img = load_image(img_path)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    return images


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> Path:
    """
    Save an image to file.
    
    Args:
        image: Numpy array representing the image
        output_path: Path to save the image
        quality: JPEG quality (1-100)
        
    Returns:
        Path to the saved image
    """
    import cv2
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in ('.jpg', '.jpeg'):
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), image)
    
    logger.debug(f"Saved image: {output_path}")
    return output_path


# ============================================================================
# JSON Serialization
# ============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and dataclasses."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(
    data: Any,
    output_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> Path:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to serialize (dict, list, dataclass, etc.)
        output_path: Path to save the JSON file
        indent: Indentation level for pretty printing
        ensure_ascii: If True, escape non-ASCII characters
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=EnhancedJSONEncoder)
    
    logger.debug(f"Saved JSON: {output_path}")
    return output_path


def load_json(json_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Directory Management
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_temp_dir(prefix: str = "doc_recon_") -> Path:
    """
    Create a temporary directory.
    
    Args:
        prefix: Prefix for the directory name
        
    Returns:
        Path to the temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug(f"Created temp directory: {temp_dir}")
    return temp_dir


def cleanup_dir(path: Union[str, Path], force: bool = False) -> bool:
    """
    Remove a directory and its contents.
    
    Args:
        path: Path to the directory
        force: If True, remove even if not a temp directory
        
    Returns:
        True if successfully removed
    """
    path = Path(path)
    if not path.exists():
        return True
    
    if not force and "doc_recon_" not in str(path):
        logger.warning(f"Refusing to delete non-temp directory: {path}")
        return False
    
    try:
        shutil.rmtree(path)
        logger.debug(f"Removed directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove directory {path}: {e}")
        return False


# ============================================================================
# File Type Detection
# ============================================================================

def detect_input_type(input_path: Union[str, Path]) -> str:
    """
    Detect the type of input file or directory.
    
    Args:
        input_path: Path to file or directory
        
    Returns:
        One of: 'pdf', 'image', 'image_folder', 'unknown'
    """
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Check if it contains images
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in input_path.iterdir()
        )
        return 'image_folder' if has_images else 'unknown'
    
    if not input_path.exists():
        return 'unknown'
    
    suffix = input_path.suffix.lower()
    if suffix == '.pdf':
        return 'pdf'
    elif suffix in ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'):
        return 'image'
    
    return 'unknown'


# ============================================================================
# Progress Tracking
# ============================================================================

@dataclass
class ProcessingProgress:
    """Track progress of document processing."""
    total_pages: int = 0
    processed_pages: int = 0
    current_stage: str = ""
    current_page: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def percent_complete(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100
    
    def update(self, stage: str, page: Optional[int] = None):
        self.current_stage = stage
        if page is not None:
            self.current_page = page
    
    def complete_page(self):
        self.processed_pages += 1
    
    def add_error(self, error: str):
        self.errors.append(error)
        logger.error(error)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Convert a PDF to images
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
        
        input_type = detect_input_type(pdf_path)
        print(f"Detected input type: {input_type}")
        
        if input_type == 'pdf':
            images = load_pdf(pdf_path, dpi=300, output_dir=output_dir)
            print(f"Converted {len(images)} pages")
        elif input_type == 'image':
            img = load_image(pdf_path)
            print(f"Loaded image with shape: {img.shape}")
        elif input_type == 'image_folder':
            images = load_images_from_folder(pdf_path)
            print(f"Loaded {len(images)} images")
    else:
        print("Usage: python io.py <input_path> [output_dir]")


