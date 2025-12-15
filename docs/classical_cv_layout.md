# Classical CV Layout Detection - Block Classification

This document explains how the **Classical CV (Computer Vision)** layout detector classifies document blocks. This method is used as the **default/fallback** when deep learning models are unavailable.

---

## Overview

The Classical CV detector uses traditional image processing techniques instead of machine learning. It's implemented in `ClassicalLayoutDetector` class in [`src/utils/layout.py`](../src/utils/layout.py).

---

## Pipeline Flow

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. GRAYSCALE CONVERSION                  │
│    cv2.cvtColor(image, COLOR_BGR2GRAY)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 2. BINARIZATION (Otsu's Threshold)       │
│    cv2.threshold(gray, THRESH_BINARY_INV │
│                       + THRESH_OTSU)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 3. MORPHOLOGICAL DILATION                │
│    - Horizontal kernel (40×1)            │
│    - Vertical kernel (1×10)              │
│    → Connects text into block regions    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 4. CONNECTED COMPONENTS ANALYSIS         │
│    cv2.connectedComponentsWithStats()    │
│    → Finds distinct regions              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 5. FILTERING                             │
│    - Remove small areas (< 100 px)       │
│    - Remove thin components              │
│      (width < 20 or height < 10)         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 6. BLOCK CLASSIFICATION (Heuristics)     │
│    See detailed rules below              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 7. MERGE OVERLAPPING BLOCKS              │
│    IoU threshold = 0.5                   │
└─────────────────────────────────────────┘
    │
    ▼
Output: List of LayoutBlock objects
```

---

## Block Classification Rules

The `_classify_block()` method uses **position and size heuristics** to determine block type:

### Computed Metrics

For each detected region, the following metrics are calculated:

```python
rel_width  = block_width / page_width      # Relative width (0-1)
rel_height = block_height / page_height    # Relative height (0-1)
rel_y      = block_y1 / page_height        # Vertical position (0=top, 1=bottom)

margin_left  = block_x1 / page_width
margin_right = (page_width - block_x2) / page_width
is_centered  = abs(margin_left - margin_right) < 0.15 AND margin_left > 0.1
```

### Classification Decision Tree

```
START
  │
  ├── rel_y < 0.15 AND rel_width > 0.4 AND rel_height < 0.08?
  │   └── YES → TITLE
  │
  ├── rel_y < 0.05?
  │   └── YES → HEADER
  │
  ├── block_y2 > page_height × 0.95?
  │   └── YES → FOOTER
  │
  ├── _looks_like_table() = True?
  │   └── YES → TABLE
  │
  ├── is_centered AND rel_height < 0.04 AND 0.15 < rel_width < 0.5 AND height < 50px?
  │   └── YES → EQUATION_BLOCK
  │
  ├── 0.5 < aspect_ratio < 2.0 AND area > 5% of page?
  │   └── YES → FIGURE
  │
  └── DEFAULT → PARAGRAPH
```

---

## Detailed Classification Criteria

### 1. TITLE
| Criterion | Threshold |
|-----------|-----------|
| Vertical position | Top 15% of page (`rel_y < 0.15`) |
| Width | > 40% of page width |
| Height | < 8% of page height |

**Rationale**: Titles are typically at the top, span most of the page width, and are short (1-2 lines).

---

### 2. HEADER
| Criterion | Threshold |
|-----------|-----------|
| Vertical position | Top 5% of page (`rel_y < 0.05`) |

**Rationale**: Page headers (running headers, page numbers) appear at the very top.

---

### 3. FOOTER
| Criterion | Threshold |
|-----------|-----------|
| Vertical position | Bottom 5% of page (`y2 > height × 0.95`) |

**Rationale**: Page footers appear at the very bottom.

---

### 4. TABLE
Detected using `_looks_like_table()` method with **morphological line detection**:

#### Bordered Tables
```python
# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(MORPH_RECT, (width//20, 1))
horizontal_lines = cv2.morphologyEx(binary, MORPH_OPEN, horizontal_kernel)

# Detect vertical lines  
vertical_kernel = cv2.getStructuringElement(MORPH_RECT, (1, height//20))
vertical_lines = cv2.morphologyEx(binary, MORPH_OPEN, vertical_kernel)

# Check for grid intersections
combined = cv2.bitwise_and(horizontal_lines, vertical_lines)
```

| Criterion | Threshold |
|-----------|-----------|
| Minimum size | height > 50px AND width > 100px |
| Horizontal lines | > 2 normalized line count |
| Vertical lines | > 2 normalized line count |
| Intersections | > 4 intersection points |

#### Borderless Tables
If no lines detected, check for **regular rectangular patterns**:
- Find contours in the region
- Count rectangular contours with reasonable cell size
- If ≥ 6 similar-sized cells found → TABLE

---

### 5. EQUATION_BLOCK
| Criterion | Threshold |
|-----------|-----------|
| Centered | Left/right margins differ by < 15% AND left margin > 10% |
| Height | < 4% of page height |
| Width | Between 15-50% of page width |
| Pixel height | < 50 pixels (single line) |

**Rationale**: Mathematical equations are typically:
- Centered on the page
- Single line (display equations)
- Not too wide or too narrow

---

### 6. FIGURE
| Criterion | Threshold |
|-----------|-----------|
| Aspect ratio | Between 0.5 and 2.0 (roughly square-ish) |
| Area | > 5% of page area |

**Rationale**: Figures/images are typically large and roughly proportioned.

---

### 7. PARAGRAPH (Default)
Any block that doesn't match the above criteria is classified as **PARAGRAPH**.

---

## Post-Processing: Block Refinement

After initial classification, the `_refine_block_types()` method in `DefaultLayoutDetector` makes additional adjustments:

```python
# Refine based on position
if block_y < page_height * 0.05:
    if block_type == PARAGRAPH:
        block_type = HEADER

if block_y > page_height * 0.95:
    if block_type == PARAGRAPH:
        block_type = FOOTER

# First large block at top becomes TITLE
if idx == 0 and block_y < page_height * 0.2 and block_width > page_width * 0.3:
    block_type = TITLE
```

---

## Confidence Scores

All blocks detected by Classical CV are assigned a **confidence of 0.7** (70%), which is lower than ML-based methods to reflect the heuristic nature of the classification.

---

## Limitations

| Limitation | Impact |
|------------|--------|
| No semantic understanding | Cannot distinguish headings from short paragraphs |
| Fixed thresholds | May not work well for non-standard layouts |
| No multi-column detection | Columns may merge into single blocks |
| Equation false positives | Centered short text may be misclassified |
| No inline equation detection | Only detects block/display equations |

---

## When to Use Classical CV

| ✅ Use When | ❌ Avoid When |
|------------|--------------|
| Simple single-column documents | Complex multi-column layouts |
| No ML dependencies available | High accuracy required |
| Fast processing needed | Academic papers with many equations |
| Scanned forms/tables | Mixed content documents |
