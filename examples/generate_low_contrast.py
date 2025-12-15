"""Generate a low contrast sample image for testing preprocessing."""

import numpy as np
import cv2
from pathlib import Path


def generate_low_contrast_sample():
    """Create a low contrast document image."""
    # Create a gray background (not white, simulating poor scan)
    width, height = 800, 1000
    img = np.ones((height, width, 3), dtype=np.uint8) * 180  # Light gray background
    
    # Use dark gray text instead of black (low contrast)
    text_color = (120, 120, 120)  # Dark gray - low contrast against light gray
    
    # Title
    cv2.putText(img, "Low Contrast Document", (150, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
    
    # Subtitle
    cv2.putText(img, "Testing CLAHE Enhancement", (200, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1)
    
    # Paragraph 1
    y = 200
    paragraph1 = [
        "This document has intentionally low contrast to test",
        "the preprocessing pipeline. The text appears gray",
        "on a gray background, making it difficult to read.",
    ]
    for line in paragraph1:
        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y += 30
    
    # Section header
    y += 20
    cv2.putText(img, "Section 1: Mathematical Content", (50, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    y += 40
    
    # Equation placeholder (centered)
    eq_text = "E = mc^2"
    text_size = cv2.getTextSize(eq_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    x_centered = (width - text_size[0]) // 2
    cv2.putText(img, eq_text, (x_centered, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    y += 50
    
    # More text
    paragraph2 = [
        "The equation above demonstrates how low contrast",
        "affects equation recognition. The CLAHE algorithm",
        "should enhance local contrast to improve OCR.",
    ]
    for line in paragraph2:
        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y += 30
    
    # Table section
    y += 30
    cv2.putText(img, "Table 1: Sample Data", (50, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    y += 30
    
    # Draw a simple table with low contrast lines
    table_color = (140, 140, 140)  # Even lower contrast for lines
    table_x, table_y = 100, y
    table_w, table_h = 400, 150
    cell_h = 50
    
    # Outer border
    cv2.rectangle(img, (table_x, table_y), (table_x + table_w, table_y + table_h), table_color, 1)
    
    # Horizontal lines
    cv2.line(img, (table_x, table_y + cell_h), (table_x + table_w, table_y + cell_h), table_color, 1)
    cv2.line(img, (table_x, table_y + 2*cell_h), (table_x + table_w, table_y + 2*cell_h), table_color, 1)
    
    # Vertical lines
    cv2.line(img, (table_x + 200, table_y), (table_x + 200, table_y + table_h), table_color, 1)
    
    # Table content
    cv2.putText(img, "Column A", (table_x + 50, table_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "Column B", (table_x + 250, table_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "Value 1", (table_x + 60, table_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "100", (table_x + 280, table_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "Value 2", (table_x + 60, table_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "200", (table_x + 280, table_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    y = table_y + table_h + 50
    
    # Final paragraph
    paragraph3 = [
        "After preprocessing with CLAHE, the contrast should",
        "be significantly improved, making all text elements",
        "more readable for the OCR engine.",
    ]
    for line in paragraph3:
        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y += 30
    
    # Add some noise to simulate scan artifacts
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def main():
    output_dir = Path(__file__).parent / "sample_pages"
    output_dir.mkdir(exist_ok=True)
    
    # Generate low contrast sample
    img = generate_low_contrast_sample()
    
    output_path = output_dir / "low_contrast.png"
    cv2.imwrite(str(output_path), img)
    print(f"Generated: {output_path}")
    
    # Also show contrast statistics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image statistics:")
    print(f"  Mean intensity: {np.mean(gray):.1f}")
    print(f"  Std deviation: {np.std(gray):.1f}")
    print(f"  Min: {np.min(gray)}, Max: {np.max(gray)}")
    print(f"  Contrast range: {np.max(gray) - np.min(gray)}")


if __name__ == "__main__":
    main()
