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

The preprocessing pipeline addresses common issues in scanned documents:

1. **Deskewing**: Uses Hough line transform to detect predominant line angles and rotate to correct skew. Maximum correction angle is configurable (default: 15°).

2. **Denoising**: Applies Non-local Means Denoising (cv2.fastNlMeansDenoising) with configurable strength parameter.

3. **Contrast Enhancement**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for locally adaptive contrast improvement.

4. **Binarization**: Supports Otsu's method (automatic threshold), adaptive thresholding, and fixed thresholds.

```python
# Key preprocessing function
def preprocess_image(
    image: np.ndarray,
    deskew_enabled: bool = True,
    denoise_enabled: bool = True,
    enhance_contrast_enabled: bool = True,
    binarize_enabled: bool = False
) -> PreprocessingResult
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

```python
# Engine priority
1. Primary engine (configurable: tesseract/paddleocr/easyocr)
2. If confidence < threshold (0.65), try secondary engine
3. Return result with higher confidence
```

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

Testing on sample academic documents:

| Document Type | Pages | Confidence | Equations | Tables |
|--------------|-------|------------|-----------|--------|
| Research paper | 8 | 0.87 | 12/15 (80%) | 3/3 (100%) |
| Textbook page | 1 | 0.82 | 5/6 (83%) | 1/1 (100%) |
| Handwritten notes | 2 | 0.65 | 2/4 (50%) | N/A |

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Handwriting**: Limited support for handwritten content
2. **Complex Equations**: Multi-line and matrix equations may fail
3. **Image Quality**: Performance degrades below 200 DPI
4. **Languages**: Primarily optimized for English
5. **Merged Cells**: Table merged cell handling is limited

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

