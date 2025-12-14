"""
Image preprocessing utilities for the document reconstruction pipeline.

Provides:
- Deskewing (rotation correction)
- Denoising (noise removal)
- Binarization (thresholding)
- Contrast enhancement
- Full preprocessing pipeline
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    image: np.ndarray
    original_shape: Tuple[int, int]
    deskew_angle: float = 0.0
    was_denoised: bool = False
    was_binarized: bool = False
    was_contrast_enhanced: bool = False
    transformations: List[str] = None
    
    def __post_init__(self):
        if self.transformations is None:
            self.transformations = []


@dataclass
class ImageStats:
    """Statistics about an image."""
    height: int
    width: int
    channels: int
    mean_intensity: float
    std_intensity: float
    estimated_dpi: Optional[int] = None
    is_grayscale: bool = False
    is_binary: bool = False


# ============================================================================
# Core Preprocessing Functions
# ============================================================================

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it's color.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    import cv2
    
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            return image.squeeze()
    
    raise ValueError(f"Unexpected image shape: {image.shape}")


def deskew(
    image: np.ndarray,
    max_angle: float = 15.0,
    return_angle: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Correct image skew by detecting and rotating to fix text alignment.
    
    Uses Hough line transform to detect predominant line angles.
    
    Args:
        image: Input image (grayscale or color)
        max_angle: Maximum angle to correct (degrees)
        return_angle: If True, also return the detected angle
        
    Returns:
        Deskewed image, or tuple of (image, angle) if return_angle=True
    """
    import cv2
    
    # Convert to grayscale for processing
    gray = to_grayscale(image) if len(image.shape) == 3 else image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        logger.debug("No lines detected for deskewing")
        return (image, 0.0) if return_angle else image
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < max_angle:
                angles.append(angle)
    
    if not angles:
        logger.debug("No valid angles found for deskewing")
        return (image, 0.0) if return_angle else image
    
    # Use median angle to be robust against outliers
    median_angle = np.median(angles)
    
    # Only correct if the angle is significant
    if abs(median_angle) < 0.5:
        logger.debug(f"Skew angle too small to correct: {median_angle:.2f}°")
        return (image, median_angle) if return_angle else image
    
    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new bounding box size
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # Apply rotation with white background
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    logger.info(f"Deskewed image by {median_angle:.2f}°")
    
    return (rotated, median_angle) if return_angle else rotated


def denoise(
    image: np.ndarray,
    strength: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Remove noise from image using Non-local Means Denoising.
    
    Args:
        image: Input image
        strength: Filter strength (higher = more denoising but more blur)
        template_window_size: Size of template patch (should be odd)
        search_window_size: Size of search area (should be odd)
        
    Returns:
        Denoised image
    """
    import cv2
    
    if len(image.shape) == 2:
        # Grayscale
        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            strength,
            template_window_size,
            search_window_size
        )
    else:
        # Color
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            strength,
            strength,
            template_window_size,
            search_window_size
        )
    
    logger.debug(f"Applied denoising with strength {strength}")
    return denoised


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    threshold: int = 0,
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Convert image to binary (black and white).
    
    Args:
        image: Input image (grayscale or color)
        method: Binarization method ('otsu', 'adaptive', 'fixed')
        threshold: Fixed threshold value (only for method='fixed')
        block_size: Block size for adaptive thresholding (must be odd)
        c: Constant subtracted for adaptive thresholding
        
    Returns:
        Binary image
    """
    import cv2
    
    # Convert to grayscale if needed
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    
    if method == "otsu":
        # Otsu's automatic thresholding
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        # Adaptive thresholding (good for varying illumination)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
    elif method == "fixed":
        # Simple fixed threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    logger.debug(f"Applied {method} binarization")
    return binary


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8
) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    import cv2
    
    if len(image.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )
        enhanced = clahe.apply(image)
    else:
        # Color - convert to LAB and enhance L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    logger.debug(f"Applied CLAHE contrast enhancement (clip={clip_limit})")
    return enhanced


def remove_borders(
    image: np.ndarray,
    border_color: Tuple[int, int, int] = (0, 0, 0),
    tolerance: int = 30
) -> np.ndarray:
    """
    Remove black or dark borders from scanned images.
    
    Args:
        image: Input image
        border_color: Color to consider as border (BGR)
        tolerance: Color tolerance for border detection
        
    Returns:
        Image with borders removed
    """
    import cv2
    
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    
    # Threshold to find dark regions
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour (assumed to be the document)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add small padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    cropped = image[y:y+h, x:x+w]
    
    logger.debug(f"Removed borders: original {image.shape[:2]} -> cropped {cropped.shape[:2]}")
    return cropped


def resize_for_ocr(
    image: np.ndarray,
    target_dpi: int = 300,
    current_dpi: Optional[int] = None
) -> np.ndarray:
    """
    Resize image to optimal resolution for OCR.
    
    Args:
        image: Input image
        target_dpi: Target DPI (300 is standard for OCR)
        current_dpi: Current DPI if known
        
    Returns:
        Resized image
    """
    import cv2
    
    if current_dpi is None:
        # Estimate current DPI based on image size
        # Assume standard letter page (8.5 x 11 inches)
        h, w = image.shape[:2]
        estimated_dpi = max(w / 8.5, h / 11)
        current_dpi = int(estimated_dpi)
    
    if current_dpi == target_dpi:
        return image
    
    scale = target_dpi / current_dpi
    
    # Don't upscale too much (diminishing returns and adds noise)
    if scale > 2.0:
        scale = 2.0
        logger.warning(f"Limiting upscale factor to 2.0 (requested would be {target_dpi/current_dpi:.1f}x)")
    
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    logger.debug(f"Resized image: {image.shape[:2]} -> {resized.shape[:2]} (scale={scale:.2f})")
    return resized


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def preprocess_image(
    image: np.ndarray,
    deskew_enabled: bool = True,
    denoise_enabled: bool = True,
    binarize_enabled: bool = False,
    enhance_contrast_enabled: bool = True,
    remove_borders_enabled: bool = False,
    denoise_strength: int = 10,
    contrast_clip_limit: float = 2.0,
    contrast_grid_size: int = 8,
    max_deskew_angle: float = 15.0,
    binarize_method: str = "otsu",
    target_dpi: Optional[int] = None
) -> PreprocessingResult:
    """
    Apply full preprocessing pipeline to an image.
    
    Args:
        image: Input image (BGR or grayscale)
        deskew_enabled: Whether to correct skew
        denoise_enabled: Whether to remove noise
        binarize_enabled: Whether to binarize (usually False for OCR)
        enhance_contrast_enabled: Whether to enhance contrast
        remove_borders_enabled: Whether to remove dark borders
        denoise_strength: Denoising filter strength
        contrast_clip_limit: CLAHE clip limit
        contrast_grid_size: CLAHE grid size
        max_deskew_angle: Maximum deskew angle
        binarize_method: Binarization method if enabled
        target_dpi: Target DPI for resizing (None = no resize)
        
    Returns:
        PreprocessingResult with processed image and metadata
    """
    original_shape = image.shape[:2]
    processed = image.copy()
    transformations = []
    deskew_angle = 0.0
    
    # 1. Remove borders if enabled
    if remove_borders_enabled:
        processed = remove_borders(processed)
        transformations.append("remove_borders")
    
    # 2. Resize for OCR if target DPI specified
    if target_dpi is not None:
        processed = resize_for_ocr(processed, target_dpi)
        transformations.append(f"resize_to_{target_dpi}dpi")
    
    # 3. Deskew
    if deskew_enabled:
        processed, deskew_angle = deskew(processed, max_deskew_angle, return_angle=True)
        if abs(deskew_angle) > 0.5:
            transformations.append(f"deskew_{deskew_angle:.1f}deg")
    
    # 4. Denoise
    if denoise_enabled:
        processed = denoise(processed, strength=denoise_strength)
        transformations.append("denoise")
    
    # 5. Enhance contrast
    if enhance_contrast_enabled:
        processed = enhance_contrast(
            processed,
            clip_limit=contrast_clip_limit,
            grid_size=contrast_grid_size
        )
        transformations.append("enhance_contrast")
    
    # 6. Binarize (usually last if enabled)
    if binarize_enabled:
        processed = binarize(processed, method=binarize_method)
        transformations.append(f"binarize_{binarize_method}")
    
    logger.info(f"Preprocessing complete: {' -> '.join(transformations) or 'no changes'}")
    
    return PreprocessingResult(
        image=processed,
        original_shape=original_shape,
        deskew_angle=deskew_angle,
        was_denoised=denoise_enabled,
        was_binarized=binarize_enabled,
        was_contrast_enhanced=enhance_contrast_enabled,
        transformations=transformations
    )


def get_image_stats(image: np.ndarray) -> ImageStats:
    """
    Calculate statistics about an image.
    
    Args:
        image: Input image
        
    Returns:
        ImageStats with image properties
    """
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    gray = to_grayscale(image) if channels > 1 else image
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    
    # Check if already binary
    unique_values = np.unique(gray)
    is_binary = len(unique_values) <= 2
    
    # Estimate DPI based on image size (assuming letter page)
    estimated_dpi = int(max(w / 8.5, h / 11))
    
    return ImageStats(
        height=h,
        width=w,
        channels=channels,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        estimated_dpi=estimated_dpi,
        is_grayscale=(channels == 1),
        is_binary=is_binary
    )


# ============================================================================
# Debug Visualization
# ============================================================================

def draw_debug_image(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image for debugging.
    
    Args:
        image: Input image
        boxes: List of (x, y, width, height) tuples
        labels: Optional labels for each box
        colors: Optional colors for each box (BGR)
        line_width: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    import cv2
    
    # Convert grayscale to color for visualization
    if len(image.shape) == 2:
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = image.copy()
    
    # Default colors cycle
    default_colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, line_width)
        
        if labels and i < len(labels):
            cv2.putText(
                debug_img,
                labels[i],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    return debug_img


def create_comparison_image(
    original: np.ndarray,
    processed: np.ndarray,
    title_original: str = "Original",
    title_processed: str = "Processed"
) -> np.ndarray:
    """
    Create a side-by-side comparison of original and processed images.
    
    Args:
        original: Original image
        processed: Processed image
        title_original: Title for original
        title_processed: Title for processed
        
    Returns:
        Combined comparison image
    """
    import cv2
    
    # Ensure both images are color
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Resize to same height
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    target_height = max(h1, h2)
    
    if h1 != target_height:
        scale = target_height / h1
        original = cv2.resize(original, (int(w1 * scale), target_height))
    
    if h2 != target_height:
        scale = target_height / h2
        processed = cv2.resize(processed, (int(w2 * scale), target_height))
    
    # Add titles
    title_height = 30
    title_bar1 = np.ones((title_height, original.shape[1], 3), dtype=np.uint8) * 255
    title_bar2 = np.ones((title_height, processed.shape[1], 3), dtype=np.uint8) * 255
    
    cv2.putText(title_bar1, title_original, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(title_bar2, title_processed, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    original_with_title = np.vstack([title_bar1, original])
    processed_with_title = np.vstack([title_bar2, processed])
    
    # Separator
    separator = np.ones((original_with_title.shape[0], 5, 3), dtype=np.uint8) * 128
    
    # Combine
    comparison = np.hstack([original_with_title, separator, processed_with_title])
    
    return comparison


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import cv2
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "preprocessed.png"
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            sys.exit(1)
        
        # Get stats
        stats = get_image_stats(image)
        print(f"Image stats: {stats}")
        
        # Preprocess
        result = preprocess_image(
            image,
            deskew_enabled=True,
            denoise_enabled=True,
            enhance_contrast_enabled=True,
            binarize_enabled=False
        )
        
        print(f"Transformations: {result.transformations}")
        print(f"Deskew angle: {result.deskew_angle:.2f}°")
        
        # Create comparison
        comparison = create_comparison_image(image, result.image)
        
        # Save
        cv2.imwrite(output_path, result.image)
        cv2.imwrite(output_path.replace('.', '_comparison.'), comparison)
        
        print(f"Saved preprocessed image to: {output_path}")
    else:
        print("Usage: python images.py <input_image> [output_image]")

