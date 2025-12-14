# Document Reconstruction Pipeline

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A modular, production-quality OCR and document reconstruction system that converts scanned academic documents into structured formats (JSON, Markdown, DOCX, PDF).

## 🌟 Features

- **PDF/Image Processing**: Convert PDFs to high-resolution images at 300-600 DPI
- **Adaptive Preprocessing Pipeline**: 
  - Auto-detects image quality (clean PDF vs degraded scan)
  - Clean images: Minimal preprocessing to preserve quality
  - Degraded images: CLAHE, adaptive thresholding, noise removal
  - Handles uneven lighting, salt-and-pepper noise, low contrast, blur
- **Automatic Layout Detection**: 
  - **Pix2Text** (recommended): Automatic equation and table detection - free, open-source!
  - LayoutParser/PaddleOCR: Deep learning-based layout analysis
  - Classical CV: Fallback method using heuristics
- **Text OCR**: Multi-engine support (Tesseract OCR, PaddleOCR, EasyOCR) with confidence scoring
- **Math/Equation OCR**: Convert equations to LaTeX using pix2tex (free) or Mathpix API (paid)
- **Table Extraction**: Reconstruct tables to Markdown, HTML, CSV formats using Camelot and image-based detection
- **Multi-format Export**: JSON, Markdown, DOCX, LaTeX, PDF outputs
- **Web UI**: Clean Streamlit interface with engine status indicators and processing feedback

## 📋 Requirements

### System Dependencies

- Python 3.10+
- Tesseract OCR
- Poppler (for PDF processing)
- LaTeX (optional, for PDF generation)

### Quick Install (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils texlive-latex-base
```

### Quick Install (macOS)

```bash
brew install tesseract poppler
brew install --cask mactex  # Optional for PDF generation
```

### Quick Install (Windows)

1. Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download and install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)
3. Add both to your PATH

## 🚀 Installation

### Using pip (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd doc_recon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Docker (easiest)

```bash
# Build the image
docker build -t doc-recon -f docker/Dockerfile .

# Run the web UI
docker run -p 8080:8080 doc-recon

# Or use docker-compose
docker-compose up
```

## 📖 Usage

### Command Line Interface

```bash
# Navigate to project directory first
cd doc_recon

# Basic usage
python src/cli.py --input document.pdf --output ./output --format all

# Process with specific options
python src/cli.py --input document.pdf --output ./output \
    --format markdown docx \
    --dpi 300 \
    --ocr-engine tesseract \
    --math-engine pix2tex

# Enable debug mode (outputs bounding box images)
python src/cli.py --input document.pdf --output ./output --debug

# Use GPU acceleration
python src/cli.py --input document.pdf --output ./output --use-gpu

# Process specific pages
python src/cli.py --input document.pdf --output ./output --pages 1-5
```

### Web Interface

```bash
# Navigate to project directory first
cd doc_recon

# Start Streamlit app
streamlit run src/app.py

# Access at http://localhost:8501
```

### Python API

```python
import sys
sys.path.insert(0, "src")  # Add src to path

from utils.io import load_pdf
from utils.assembler import DocumentAssembler
from utils.export import DocumentExporter

# Load PDF
images = load_pdf("document.pdf", dpi=300)

# Process document
assembler = DocumentAssembler(use_gpu=False, output_dir="./output")
document = assembler.process_document(images, source_file="document.pdf")

# Export to multiple formats
exporter = DocumentExporter("./output", "document")
exporter.export(document, formats=["markdown", "docx", "json"])

# Access results
print(f"Confidence: {document.metrics.global_confidence:.1%}")
print(f"Markdown:\n{document.markdown}")
```

## 📁 Project Structure

```
doc_recon/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── docker/
│   └── Dockerfile          # Docker build file
├── docker-compose.yml      # Docker Compose config
├── src/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── app.py              # Streamlit web app
│   ├── config.py           # Configuration and constants
│   ├── utils/
│   │   ├── io.py           # File I/O utilities
│   │   ├── images.py       # Image preprocessing
│   │   ├── layout.py       # Layout detection
│   │   ├── ocr_text.py     # Text OCR
│   │   ├── ocr_math.py     # Math/equation OCR
│   │   ├── tables.py       # Table extraction
│   │   ├── assembler.py    # Document assembly
│   │   └── export.py       # Multi-format export
│   └── models/             # Model weights (gitignored)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_layout.py
│   ├── test_ocr_text.py
│   ├── test_ocr_math.py
│   └── test_end2end.py
├── examples/
│   ├── sample_pages/       # Sample input documents
│   └── expected_outputs/   # Expected outputs for validation
├── docs/
│   └── project_report.md   # Project documentation
└── eval.py                 # Evaluation script
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run end-to-end tests
pytest tests/test_end2end.py -v
```

## 📊 Evaluation

```bash
# Evaluate a single document
python eval.py --input output/document.json

# Evaluate against expected output
python eval.py --input output/document.json --expected examples/expected_outputs/document.json

# Evaluate a directory of results
python eval.py --results-dir ./output --report report.json
```

### Metrics

The pipeline reports the following metrics:

| Metric | Description |
|--------|-------------|
| `global_confidence` | Average confidence across all blocks |
| `coverage_pct` | Percentage of page area covered by detected blocks |
| `text_blocks_total` | Number of detected text blocks |
| `equations_total` | Number of detected equations |
| `tables_total` | Number of detected tables |
| `equation_success_rate` | Percentage of equations with valid LaTeX |
| `table_success_rate` | Percentage of tables fully reconstructed |

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOC_RECON_USE_GPU` | Enable GPU acceleration | `false` |
| `DOC_RECON_DEBUG` | Enable debug mode | `false` |
| `MATHPIX_APP_ID` | Mathpix API ID (optional) | - |
| `MATHPIX_APP_KEY` | Mathpix API Key (optional) | - |

### Optional Libraries for Better Detection

#### Pix2Text (Recommended for Automatic Equation/Table Detection)

**Pix2Text** is a free, open-source library that automatically detects equations and tables in documents. It's the easiest way to get good results!

```bash
pip install pix2text
```

**Benefits:**
- ✅ Automatic equation detection and LaTeX conversion
- ✅ Automatic table detection and structure extraction
- ✅ Free and open-source (no API keys needed)
- ✅ Works offline
- ✅ Supports 80+ languages

The system will automatically use Pix2Text if installed, otherwise falls back to other methods.

#### Mathpix (for improved equation recognition - Paid)

1. Create an account at [mathpix.com](https://mathpix.com)
2. Get your API credentials (free tier: 1,000 calls/month)
3. Set environment variables:

```bash
export MATHPIX_APP_ID="your_app_id"
export MATHPIX_APP_KEY="your_app_key"
```

## 📤 Output Format

### JSON Schema

```json
{
  "task_id": "uuid",
  "schema_version": "1.0",
  "source_file": "document.pdf",
  "created_at": "ISO8601 timestamp",
  "metadata": {
    "title": "Document Title",
    "authors": "Author Names",
    "abstract": "Abstract text"
  },
  "pages": [
    {
      "page_number": 1,
      "width": 2550,
      "height": 3300,
      "num_columns": 1,
      "blocks": [
        {
          "block_id": "abc123",
          "type": "paragraph",
          "bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 250},
          "reading_order": 0,
          "confidence": 0.95,
          "text": "Extracted text content"
        }
      ]
    }
  ],
  "metrics": {
    "global_confidence": 0.89,
    "coverage_pct": 75.5,
    "text_blocks": {"total": 15, "high_confidence": 12, "low_confidence": 1},
    "equations": {"total": 3, "high_confidence": 2, "low_confidence": 1},
    "tables": {"total": 1, "reconstructed": 1, "partial": 0}
  },
  "markdown": "# Document Title\n\nContent...",
  "docx_manifest": {...}
}
```

## 🔧 Troubleshooting

### Common Issues

**Tesseract not found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify installation
tesseract --version
```

**Poppler not found (PDF processing fails)**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows: Download from GitHub releases and add to PATH
```

**GPU not detected**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Low OCR accuracy**
- Ensure input images are at least 300 DPI (400+ recommended for academic docs)
- The system auto-detects image quality and applies appropriate preprocessing:
  - Clean PDFs/screenshots: Minimal processing (aggressive preprocessing hurts clean images)
  - Degraded scans: CLAHE + adaptive threshold + noise removal
- Try different OCR engines:
  - Tesseract OCR: Best for clean, high-DPI academic documents
  - PaddleOCR: Better for degraded images and non-English text
  - EasyOCR: Good multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [LayoutParser](https://github.com/Layout-Parser/layout-parser)
- [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR)
- [Camelot](https://github.com/camelot-dev/camelot)
- [Streamlit](https://streamlit.io/)

