#!/usr/bin/env python
"""
Command-line interface for the Document Reconstruction Pipeline.

Usage:
    python src/cli.py --input <pdf_or_image> --output <output_dir> [options]
    
Examples:
    # Process a PDF
    python src/cli.py --input document.pdf --output ./output --format all
    
    # Process with GPU
    python src/cli.py --input document.pdf --output ./output --use-gpu
    
    # Debug mode with bounding box visualization
    python src/cli.py --input document.pdf --output ./output --debug
"""

import sys
from pathlib import Path

# Add src directory to path for imports when running as script
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import argparse
import logging
import json
import time
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("doc_recon")


def setup_argparser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Document Reconstruction Pipeline - Convert scanned documents to structured formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a PDF and export all formats:
    python -m src.cli --input document.pdf --output ./output --format all
    
  Process with GPU acceleration:
    python -m src.cli --input document.pdf --output ./output --use-gpu
    
  Debug mode with bounding box visualization:
    python -m src.cli --input document.pdf --output ./output --debug
    
  Process only specific pages:
    python -m src.cli --input document.pdf --output ./output --pages 1-5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PDF file or folder of images"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for generated files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--format", "-f",
        nargs="+",
        default=["json", "markdown"],
        choices=["json", "markdown", "docx", "latex", "pdf", "all"],
        help="Output format(s) (default: json markdown)"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for model inference if available"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (outputs debug images with bounding boxes)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)"
    )
    
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to process, e.g., '1-5' or '1,3,5' (default: all)"
    )
    
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "paddleocr", "easyocr"],
        default="tesseract",
        help="Primary OCR engine (default: tesseract)"
    )
    
    parser.add_argument(
        "--math-engine",
        choices=["pix2tex", "mathpix", "simple"],
        default="pix2tex",
        help="Math OCR engine (default: pix2tex)"
    )
    
    parser.add_argument(
        "--layout-method",
        choices=["auto", "classical", "layoutparser", "paddleocr", "pix2text"],
        default="auto",
        help="Layout detection method (default: auto - tries best available)"
    )
    
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable image preprocessing (deskew, denoise)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum confidence threshold for OCR results (default: 0.65)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser


def parse_page_range(page_str: str, max_pages: int) -> List[int]:
    """Parse page range string to list of page numbers."""
    pages = []
    
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start = int(start)
            end = min(int(end), max_pages)
            pages.extend(range(start, end + 1))
        else:
            page = int(part)
            if 1 <= page <= max_pages:
                pages.append(page)
    
    return sorted(set(pages))


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    optional_missing = []
    
    # Required
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pytesseract
        # Test if tesseract is actually installed
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            missing.append("tesseract-ocr (system package)")
    except ImportError:
        missing.append("pytesseract")
    
    # PDF support
    try:
        import pdf2image
    except ImportError:
        optional_missing.append("pdf2image (for PDF support)")
    
    # Optional - deep learning
    try:
        import torch
    except ImportError:
        optional_missing.append("pytorch (for GPU acceleration)")
    
    try:
        from pix2tex.cli import LatexOCR
    except ImportError:
        optional_missing.append("pix2tex (for equation recognition)")
    
    # Report
    if missing:
        logger.error("Missing required dependencies:")
        for dep in missing:
            logger.error(f"  - {dep}")
        logger.error("\nInstall with: pip install -r requirements.txt")
        return False
    
    if optional_missing:
        logger.warning("Missing optional dependencies (some features may be limited):")
        for dep in optional_missing:
            logger.warning(f"  - {dep}")
    
    return True


def run_pipeline(args) -> int:
    """Run the document reconstruction pipeline."""
    from utils.io import load_pdf, load_image, load_images_from_folder
    from utils.io import detect_input_type, save_json, ensure_dir
    from utils.assembler import DocumentAssembler
    from utils.export import DocumentExporter
    from config import get_config, check_gpu_available
    
    start_time = time.time()
    
    # Setup output directory
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    
    # Check GPU
    if args.use_gpu:
        if check_gpu_available():
            logger.info("GPU acceleration enabled")
        else:
            logger.warning("GPU requested but not available, using CPU")
            args.use_gpu = False
    
    # Detect input type and load images
    input_path = Path(args.input)
    input_type = detect_input_type(input_path)
    
    logger.info(f"Input type detected: {input_type}")
    
    if input_type == "pdf":
        logger.info(f"Converting PDF to images at {args.dpi} DPI...")
        images = load_pdf(input_path, dpi=args.dpi)
        pdf_path = str(input_path)
    elif input_type == "image":
        logger.info("Loading single image...")
        images = [load_image(input_path)]
        pdf_path = None
    elif input_type == "image_folder":
        logger.info("Loading images from folder...")
        images = load_images_from_folder(input_path)
        pdf_path = None
    else:
        logger.error(f"Unsupported input type: {input_type}")
        return 1
    
    if not images:
        logger.error("No images to process")
        return 1
    
    logger.info(f"Loaded {len(images)} page(s)")
    
    # Filter pages if specified
    if args.pages:
        page_indices = parse_page_range(args.pages, len(images))
        images = [images[i-1] for i in page_indices if i-1 < len(images)]
        logger.info(f"Processing pages: {page_indices}")
    
    # Create assembler
    assembler = DocumentAssembler(
        use_gpu=args.use_gpu,
        debug_mode=args.debug,
        output_dir=output_dir,
        ocr_engine=args.ocr_engine,
        math_engine=args.math_engine,
        layout_method=args.layout_method
    )
    
    # Process document
    logger.info("Processing document...")
    try:
        document = assembler.process_document(
            images,
            source_file=str(input_path),
            pdf_path=pdf_path
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.debug:
            raise
        return 1
    
    # Save JSON output
    json_path = output_dir / "document.json"
    save_json(document.to_dict(), json_path)
    logger.info(f"Saved JSON: {json_path}")
    
    # Export to requested formats
    formats = args.format
    if "all" in formats:
        formats = ["markdown", "docx", "latex", "pdf"]
    
    if any(f in formats for f in ["markdown", "docx", "latex", "pdf"]):
        exporter = DocumentExporter(output_dir, input_path.stem)
        export_results = exporter.export(document, formats)
        
        for fmt, path in export_results.items():
            logger.info(f"Exported {fmt}: {path}")
    
    # Print summary
    elapsed = time.time() - start_time
    metrics = document.metrics
    
    if not args.quiet:
        print("\n" + "="*60)
        print("DOCUMENT RECONSTRUCTION COMPLETE")
        print("="*60)
        print(f"Source: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Pages processed: {metrics.pages_processed}")
        print(f"Processing time: {elapsed:.2f}s")
        print()
        print("Metrics:")
        print(f"  Global confidence: {metrics.global_confidence:.2%}")
        print(f"  Page coverage: {metrics.coverage_pct:.1f}%")
        print(f"  Text blocks: {metrics.text_blocks_total} "
              f"(high: {metrics.text_blocks_high_confidence}, "
              f"low: {metrics.text_blocks_low_confidence})")
        print(f"  Equations: {metrics.equations_total} "
              f"(high: {metrics.equations_high_confidence}, "
              f"low: {metrics.equations_low_confidence})")
        print(f"  Tables: {metrics.tables_total} "
              f"(reconstructed: {metrics.tables_reconstructed}, "
              f"partial: {metrics.tables_partial})")
        print("="*60)
    
    return 0


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run pipeline
    try:
        exit_code = run_pipeline(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

