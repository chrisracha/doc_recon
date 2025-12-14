# Document Reconstruction Pipeline - Project Report

**Course:** CMSC 162 - Digital Image Processing  
**Project:** End-to-End OCR and Document Reconstruction System  
**Version:** 1.0

---

## Executive Summary

This project implements a comprehensive document reconstruction pipeline that converts scanned academic documents (PDFs and images) into structured, editable formats. The system employs a modular architecture combining classical computer vision techniques with modern deep learning approaches for layout detection, text recognition, equation parsing, and table extraction.

---

## 1. Introduction

### 1.1 Problem Statement

Academic documents often exist only as scanned PDFs or images, making them difficult to search, edit, or repurpose. Manual transcription is time-consuming and error-prone. This project addresses the challenge of automatically converting these documents into structured formats while preserving:

- Document structure (titles, headings, paragraphs)
- Mathematical equations (as LaTeX)
- Tables (as structured data)
- Reading order (especially for multi-column layouts)

### 1.2 Objectives

1. Build a modular, testable document processing pipeline
2. Support multiple input formats (PDF, images)
3. Generate structured outputs (JSON, Markdown, DOCX, PDF)
4. Provide confidence scores for all extracted content
5. Handle complex layouts including multi-column documents
6. Create both CLI and web-based interfaces

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDF/Image  │────▶│ Preprocessing│────▶│   Layout     │
│    Input     │     │   (images)   │     │  Detection   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
        ┌────────────────────────────────────────┴────────────────────────┐
        │                                                                 │
        ▼                              ▼                                  ▼
┌──────────────┐              ┌──────────────┐                  ┌──────────────┐
│   Text OCR   │              │   Math OCR   │                  │    Table     │
│  (blocks)    │              │  (equations) │                  │  Extraction  │
└──────────────┘              └──────────────┘                  └──────────────┘
        │                              │                                  │
        └──────────────────────────────┴──────────────────────────────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │   Document       │
                            │   Assembler      │
                            └──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             ┌──────────┐       ┌──────────┐       ┌──────────┐
             │   JSON   │       │ Markdown │       │   DOCX   │
             └──────────┘       └──────────┘       └──────────┘
```

### 2.2 Module Descriptions

| Module | Purpose | Key Technologies |
|--------|---------|------------------|
| `io.py` | File loading and saving | pdf2image, OpenCV |
| `images.py` | Image preprocessing | OpenCV, scikit-image |
| `layout.py` | Block detection and classification | LayoutParser, classical CV |
| `ocr_text.py` | Text extraction | Tesseract, PaddleOCR, EasyOCR |
| `ocr_math.py` | Equation to LaTeX | pix2tex, Mathpix API |
| `tables.py` | Table extraction | Camelot, OpenCV |
| `assembler.py` | Document assembly | Custom |
| `export.py` | Multi-format export | python-docx, LaTeX |

---

## 3. Implementation Details

### 3.1 Image Preprocessing

The preprocessing pipeline uses **adaptive processing** based on automatic image quality detection:

#### Image Quality Detection

The system automatically classifies images as "clean" or "degraded" using:
- **Laplacian variance**: Measures image sharpness (500-50000 = clean, <500 = blurry, >50000 = noisy)
- **Lighting uniformity**: Variance of local means (<30 = uniform lighting)

#### Preprocessing Modes

**Clean Images (PDFs, screenshots):**
- Minimal preprocessing to preserve quality
- Grayscale conversion and optional resizing only
- Aggressive preprocessing (adaptive threshold) destroys fine text details

**Degraded Images (scans, photos):**
1. **Median blur**: Removes salt-and-pepper noise
2. **CLAHE**: Fixes uneven lighting and low contrast
3. **Adaptive thresholding**: Binarizes for better text/background separation
4. **Morphological opening**: Cleans remaining noise artifacts

#### Global Preprocessing
1. **Deskewing**: Hough line transform to detect and correct skew (max 15°)
2. **Denoising**: Non-local Means Denoising (cv2.fastNlMeansDenoising)
3. **Contrast Enhancement**: CLAHE for locally adaptive contrast improvement

```python
# Adaptive preprocessing in TesseractEngine
def _is_clean_image(self, image) -> bool:
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_sharp = 500 < laplacian_var < 50000
    is_uniform_lighting = lighting_variance < 30
    return is_sharp and is_uniform_lighting

def _preprocess_for_ocr(self, image, aggressive: bool = False):
    if aggressive:  # For degraded images only
        gray = cv2.medianBlur(gray, 3)  # Remove noise first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)  # Fix lighting
        gray = cv2.adaptiveThreshold(...)  # Binarize
```

### 3.2 Layout Detection

Layout detection identifies and classifies document regions:

**Block Types:**
- Title, Authors, Abstract
- Headings, Paragraphs
- Equations (block and inline)
- Tables, Figures, Captions
- Headers, Footers, References

**Detection Methods:**
1. **Primary**: LayoutParser with Detectron2 (PubLayNet pretrained model)
2. **Alternative**: PaddleOCR layout detection
3. **Fallback**: Classical CV using connected components and morphological operations

**Reading Order Resolution:**
- Column detection using block center clustering
- Column-major ordering (top-to-bottom within columns)
- Multi-column support (2+ columns)

### 3.3 Text OCR

Multi-engine approach with confidence-based fallback:

**Available Engines:**
- **Tesseract OCR**: Best for clean, high-DPI documents (default)
- **PaddleOCR**: Better for degraded images, supports PaddleOCR 3.x API
- **EasyOCR**: Good multi-language support

```python
# Engine priority
1. Primary engine (configurable: Tesseract OCR/PaddleOCR/EasyOCR)
2. If confidence < threshold (0.65), try secondary engine
3. Return result with higher confidence
```

**Adaptive Preprocessing per Engine:**
- Auto-detects if image is clean (PDF) vs degraded (scan)
- Clean images: Skip aggressive preprocessing (preserves quality)
- Degraded images: Apply CLAHE + adaptive threshold + noise removal

**Post-processing:**
- Hyphenation fix: `docu-\nment` → `document`
- Line merging: Combines lines within paragraphs
- Preserves paragraph breaks

### 3.4 Math/Equation OCR

**Primary Engine: pix2tex (LaTeX-OCR)**
- Local inference using PyTorch
- No API calls required
- Works offline

**Fallback: Mathpix API**
- Optional, requires API credentials
- Better for degraded images
- Provides confidence scores

**Confidence Estimation:**
- Based on LaTeX validity (balanced braces)
- Length heuristics
- Presence of known math commands

### 3.5 Table Extraction

**Three-tier approach:**

1. **Camelot (PDF)**: For PDFs with vector tables
   - Lattice mode: Detects table lines
   - Stream mode: For borderless tables

2. **Deep Learning**: Table Transformer for table detection
   - Object detection for table regions
   - Combined with cell extraction

3. **OpenCV Fallback**: Morphological line detection
   - Horizontal/vertical kernel filtering
   - Connected component analysis
   - Grid reconstruction

**Output Formats:**
- `table_markdown`: Markdown table syntax
- `table_html`: HTML `<table>` markup
- `table_csv`: CSV string
- `table_struct`: 2D array structure

---

## 4. Data Structures

### 4.1 Core Classes

```python
@dataclass
class Block:
    bbox: BoundingBox
    block_type: BlockType
    confidence: float
    text: Optional[str] = None
    reading_order: int = 0
    column_index: int = 0

@dataclass
class Page:
    page_number: int
    width: int
    height: int
    blocks: List[ProcessedBlock]
    num_columns: int = 1

@dataclass
class Document:
    task_id: str
    source_file: str
    pages: List[Page]
    metrics: DocumentMetrics
    markdown: str
    docx_manifest: Dict[str, Any]
```

### 4.2 JSON Output Schema

```json
{
  "task_id": "string (UUID)",
  "schema_version": "1.0",
  "source_file": "string",
  "created_at": "ISO8601 datetime",
  "metadata": {
    "title": "string | 'UNREADABLE'",
    "authors": "string | 'UNREADABLE'",
    "abstract": "string | null"
  },
  "pages": [{
    "page_number": "integer",
    "blocks": [{
      "block_id": "string",
      "type": "string (enum)",
      "bbox": {"x1", "y1", "x2", "y2"},
      "confidence": "float [0,1]",
      "text": "string | null",
      "latex": "string | null",
      "table": "object | null"
    }]
  }],
  "metrics": {
    "global_confidence": "float",
    "coverage_pct": "float",
    "text_blocks": {"total", "high_confidence", "low_confidence"},
    "equations": {"total", "high_confidence", "low_confidence"},
    "tables": {"total", "reconstructed", "partial"}
  }
}
```

---

## 5. Mapping to Course CLOs

| CLO | Description | Implementation |
|-----|-------------|----------------|
| 1 | Understanding digital image fundamentals | Image loading, color space conversion, DPI handling |
| 2 | Image enhancement techniques | CLAHE contrast enhancement, denoising, binarization |
| 3 | Geometric transformations | Deskewing via rotation matrix transformation |
| 4 | Morphological operations | Table line detection, text block grouping |
| 5 | Edge and contour detection | Layout detection fallback, table grid detection |
| 6 | Image segmentation | Block extraction, cell segmentation |
| 7 | Feature extraction | OCR region features, equation boundary detection |
| 8 | Pattern recognition | Text recognition, layout classification |

---

## 6. Evaluation

### 6.1 Metrics Definition

| Metric | Formula | Target |
|--------|---------|--------|
| Global Confidence | `mean(all block confidences)` | ≥ 0.75 |
| Coverage | `(detected block area / page area) × 100` | ≥ 70% |
| Text High Confidence | `count(conf ≥ 0.80) / total` | ≥ 80% |
| Equation Success | `count(valid LaTeX) / total equations` | ≥ 70% |
| Table Reconstruction | `count(fully reconstructed) / total tables` | ≥ 60% |

### 6.2 Test Results

Testing on sample pages in `examples/sample_pages/`:

| Document | Detection | Confidence | Notes |
|----------|-----------|------------|-------|
| screenshot.png (academic PDF) | CLEAN | 89.8% | High-quality, no preprocessing needed |
| sample_math.png | CLEAN | 85.7% | Math equations detected well |
| sample_multicol.png | CLEAN | 91.5% | Multi-column layout handled |
| sample_table.png | CLEAN | 80.3% | Table structure preserved |
| blur.png (degraded scan) | DEGRADED | 34.2% | Auto-applies aggressive preprocessing |

**Adaptive Preprocessing Validation:**

| Scenario | Laplacian | Lighting Var | Detection | Result |
|----------|-----------|--------------|-----------|--------|
| Clean PDF | 5000-7000 | <10 | CLEAN | Perfect OCR |
| Blurry scan | <500 | <30 | DEGRADED | Improved with preprocessing |
| Heavy noise | >50000 | varies | DEGRADED | Noise removal applied |
| Uneven lighting | varies | >30 | DEGRADED | CLAHE fixes contrast |

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Handwriting**: Limited support for handwritten content
2. **Complex Equations**: Multi-line and matrix equations may fail
3. **Image Quality**: Performance degrades below 200 DPI (300+ recommended)
4. **Languages**: Primarily optimized for English (PaddleOCR supports more languages)
5. **Merged Cells**: Table merged cell handling is limited
6. **Extremely Degraded Images**: Very blurry or heavily damaged documents may still have low confidence despite adaptive preprocessing

### 7.2 Future Improvements

1. **Fine-tuning**: Train layout model on academic document dataset
2. **Handwriting OCR**: Integrate handwriting-specific models
3. **Multi-language**: Add language detection and models
4. **Equation Improvement**: Handle multi-line equations
5. **GPU Optimization**: Better batch processing for GPU inference
6. **Active Learning**: User feedback loop for model improvement

---

## 8. Conclusion

This project successfully implements a modular, extensible document reconstruction pipeline that handles the core requirements of converting scanned academic documents to structured formats. The system achieves good accuracy on printed documents and provides confidence scores to identify uncertain extractions.

Key achievements:
- End-to-end pipeline from PDF/image to multiple output formats
- Modular architecture allowing component replacement
- Confidence-based quality assessment
- Both CLI and web interfaces
- Docker containerization for easy deployment

The pipeline serves as a foundation that can be extended with domain-specific models and additional language support.

---

## References

1. Zhong, X., Tang, J., & Yepes, A. J. (2019). PubLayNet: Largest dataset ever for document layout analysis. *ICDAR 2019*.
2. Blecher, L., et al. (2021). pix2tex: LaTeX OCR. GitHub repository.
3. Smith, R. (2007). An overview of the Tesseract OCR engine. *ICDAR 2007*.
4. Shafait, F., et al. (2008). Document image denoising using spectral domains. *DRR 2008*.
5. Shen, Z., et al. (2021). LayoutParser: A unified toolkit for deep learning based document image analysis. *ICDAR 2021*.

