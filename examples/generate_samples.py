#!/usr/bin/env python
"""
Generate synthetic sample pages for testing the document reconstruction pipeline.

This script creates sample document images with:
- Text paragraphs
- Mathematical equations
- Tables
- Multi-column layouts

Usage:
    python examples/generate_samples.py
"""

import numpy as np
import json
from pathlib import Path

def create_sample_math_page():
    """Create a sample page with mathematical equations."""
    import cv2
    
    # Create white page (Letter size at 300 DPI: 2550 x 3300)
    # Using smaller size for faster processing: 850 x 1100
    img = np.ones((1100, 850, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(img, "Mathematical Equations Demo", (150, 60),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    
    # Introduction paragraph
    y = 120
    lines = [
        "This document demonstrates equation recognition.",
        "The following equations should be recognized:"
    ]
    for line in lines:
        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 25
    
    # Equation 1: Simple
    y += 30
    cv2.putText(img, "Equation 1 (Einstein's mass-energy):", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y += 40
    cv2.putText(img, "E = mc^2", (350, y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Equation 2: Quadratic
    y += 60
    cv2.putText(img, "Equation 2 (Quadratic formula):", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y += 40
    cv2.putText(img, "x = (-b +/- sqrt(b^2-4ac)) / 2a", (200, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Equation 3: Euler's identity
    y += 60
    cv2.putText(img, "Equation 3 (Euler's identity):", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y += 40
    cv2.putText(img, "e^(i*pi) + 1 = 0", (300, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Additional text with inline math
    y += 80
    cv2.putText(img, "Inline example: The area of a circle is A = pi*r^2", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Conclusion
    y += 60
    cv2.putText(img, "These equations demonstrate the OCR pipeline's", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y += 25
    cv2.putText(img, "ability to recognize mathematical notation.", (50, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def create_sample_table_page():
    """Create a sample page with tables."""
    import cv2
    
    img = np.ones((1100, 850, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(img, "Table Extraction Demo", (250, 60),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    
    # Introduction
    cv2.putText(img, "This page demonstrates table extraction:", (50, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Table 1: Simple 3x4 table
    table_x = 100
    table_y = 180
    cell_w = 150
    cell_h = 40
    rows = 4
    cols = 4
    
    cv2.putText(img, "Table 1: Sample Data", (50, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Headers
    headers = ["ID", "Name", "Value", "Status"]
    data = [
        ["1", "Alpha", "100", "Active"],
        ["2", "Beta", "200", "Pending"],
        ["3", "Gamma", "300", "Complete"]
    ]
    
    # Draw table
    for row in range(rows):
        for col in range(cols):
            x = table_x + col * cell_w
            y = table_y + row * cell_h
            
            # Draw cell border
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (0, 0, 0), 1)
            
            # Add text
            if row == 0:
                text = headers[col]
                cv2.putText(img, text, (x + 30, y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                text = data[row - 1][col]
                cv2.putText(img, text, (x + 30, y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Table 2: Smaller 2x2 table
    table_y = 400
    cv2.putText(img, "Table 2: Summary", (50, 380),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    summary = [["Metric", "Value"], ["Total", "600"]]
    
    for row in range(2):
        for col in range(2):
            x = table_x + col * cell_w
            y = table_y + row * cell_h
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (0, 0, 0), 1)
            cv2.putText(img, summary[row][col], (x + 30, y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Description
    cv2.putText(img, "The tables above should be extracted as structured data.", 
               (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def create_sample_multicol_page():
    """Create a sample two-column page."""
    import cv2
    
    img = np.ones((1100, 850, 3), dtype=np.uint8) * 255
    
    # Title (full width)
    cv2.putText(img, "Multi-Column Layout Demo", (200, 60),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    
    # Abstract (full width)
    cv2.putText(img, "Abstract", (50, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    abstract = "This demonstrates a two-column academic paper layout."
    cv2.putText(img, abstract, (50, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Column separator line (visual guide)
    col_mid = 425
    cv2.line(img, (col_mid, 180), (col_mid, 1050), (200, 200, 200), 1)
    
    # Left column
    left_x = 30
    y = 200
    
    cv2.putText(img, "1. Introduction", (left_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y += 30
    
    left_text = [
        "This is the first column of",
        "a two-column layout. The text",
        "here should be read before",
        "moving to the right column.",
        "",
        "Multi-column layouts are",
        "common in academic papers",
        "and require special handling",
        "to determine reading order.",
        "",
        "The pipeline should detect",
        "the column boundary and",
        "order blocks appropriately."
    ]
    
    for line in left_text:
        cv2.putText(img, line, (left_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y += 22
    
    # Right column
    right_x = 450
    y = 200
    
    cv2.putText(img, "2. Methods", (right_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y += 30
    
    right_text = [
        "This is the second column.",
        "Content here follows the",
        "first column in reading",
        "order.",
        "",
        "The layout detection module",
        "uses center-of-mass analysis",
        "to identify columns and",
        "assign reading order.",
        "",
        "2.1 Sub-section",
        "Subsections are indented",
        "and should maintain their",
        "hierarchical structure."
    ]
    
    for line in right_text:
        cv2.putText(img, line, (right_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y += 22
    
    # Footer
    cv2.putText(img, "Page 1", (400, 1080),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return img


def create_expected_output(filename: str, description: str) -> dict:
    """Create a template expected output structure."""
    return {
        "task_id": f"sample-{filename}",
        "schema_version": "1.0",
        "source_file": f"{filename}.png",
        "metadata": {
            "title": description,
            "authors": "Sample Generator",
            "abstract": None
        },
        "pages": [{
            "page_number": 1,
            "blocks": []  # Would be filled with actual detected blocks
        }],
        "notes": "This is a template - actual expected values should be filled manually"
    }


def main():
    import cv2
    
    # Create output directories
    samples_dir = Path(__file__).parent / "sample_pages"
    expected_dir = Path(__file__).parent / "expected_outputs"
    samples_dir.mkdir(exist_ok=True)
    expected_dir.mkdir(exist_ok=True)
    
    # Generate samples
    samples = [
        ("sample_math", create_sample_math_page(), "Mathematical Equations Demo"),
        ("sample_table", create_sample_table_page(), "Table Extraction Demo"),
        ("sample_multicol", create_sample_multicol_page(), "Multi-Column Layout Demo"),
    ]
    
    for name, img, description in samples:
        # Save image
        img_path = samples_dir / f"{name}.png"
        cv2.imwrite(str(img_path), img)
        print(f"Created: {img_path}")
        
        # Create expected output template
        expected = create_expected_output(name, description)
        expected_path = expected_dir / f"{name}.json"
        with open(expected_path, 'w') as f:
            json.dump(expected, f, indent=2)
        print(f"Created: {expected_path}")
    
    print("\nSample generation complete!")
    print("Note: Expected outputs are templates - fill in actual expected values for validation.")


if __name__ == "__main__":
    main()

