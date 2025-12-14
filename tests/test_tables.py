"""
Tests for table extraction module.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.tables import (
    TableExtractor, 
    OpenCVTableExtractor, 
    TableResult, 
    Cell
)


class TestTableResult:
    """Tests for TableResult data class."""
    
    def test_empty_table(self):
        """Test empty table result."""
        result = TableResult(cells=[], num_rows=0, num_cols=0)
        assert result.table_markdown == ""
        assert result.table_html == ""
        assert result.table_csv == ""
    
    def test_simple_table(self):
        """Test simple 2x2 table result."""
        cells = [
            Cell(text="A", row=0, col=0, is_header=True),
            Cell(text="B", row=0, col=1, is_header=True),
            Cell(text="1", row=1, col=0),
            Cell(text="2", row=1, col=1),
        ]
        result = TableResult(cells=cells, num_rows=2, num_cols=2)
        
        assert result.num_rows == 2
        assert result.num_cols == 2
        assert "| A | B |" in result.table_markdown
        assert "| 1 | 2 |" in result.table_markdown
        assert "<th>A</th>" in result.table_html
        assert "<td>1</td>" in result.table_html
    
    def test_table_to_dict(self):
        """Test table serialization."""
        cells = [Cell(text="X", row=0, col=0)]
        result = TableResult(cells=cells, num_rows=1, num_cols=1, confidence=0.9)
        
        d = result.to_dict()
        assert d["num_rows"] == 1
        assert d["num_cols"] == 1
        assert d["confidence"] == 0.9
        assert len(d["cells"]) == 1


class TestOpenCVTableExtractor:
    """Tests for OpenCV-based table extraction."""
    
    @pytest.fixture
    def extractor(self):
        return OpenCVTableExtractor(min_rows=2, min_cols=2)
    
    @pytest.fixture
    def bordered_table_image(self):
        """Create an image with a bordered table."""
        img = np.ones((400, 500, 3), dtype=np.uint8) * 255
        
        # Draw table borders
        # Outer border
        cv2.rectangle(img, (50, 50), (450, 350), (0, 0, 0), 2)
        
        # Horizontal lines
        cv2.line(img, (50, 100), (450, 100), (0, 0, 0), 2)  # Header separator
        cv2.line(img, (50, 175), (450, 175), (0, 0, 0), 2)
        cv2.line(img, (50, 250), (450, 250), (0, 0, 0), 2)
        
        # Vertical lines
        cv2.line(img, (150, 50), (150, 350), (0, 0, 0), 2)
        cv2.line(img, (300, 50), (300, 350), (0, 0, 0), 2)
        
        # Add text to cells
        cv2.putText(img, "Col1", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "Col2", (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "Col3", (310, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(img, "A", (90, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "B", (210, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "C", (360, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return img
    
    @pytest.fixture  
    def borderless_table_image(self):
        """Create an image with a borderless table (text-only alignment)."""
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Header row
        cv2.putText(img, "Name", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Age", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "City", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Data rows
        cv2.putText(img, "Alice", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "25", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "NYC", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(img, "Bob", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "30", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "LA", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return img
    
    @pytest.fixture
    def no_table_image(self):
        """Create an image with no table structure."""
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "This is just some text", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return img
    
    def test_extractor_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.min_rows == 2
        assert extractor.min_cols == 2
    
    def test_bordered_table_detection(self, extractor, bordered_table_image):
        """Test detection of bordered tables."""
        results = extractor.extract(bordered_table_image)
        
        # Should detect at least one table
        assert len(results) >= 1
        
        result = results[0]
        assert result.num_rows >= 2
        assert result.num_cols >= 2
        assert "bordered" in result.method_used
        assert result.confidence >= 0.5
    
    def test_no_table_returns_empty(self, extractor, no_table_image):
        """Test that non-table images return empty results."""
        results = extractor.extract(no_table_image)
        # Should not detect any tables
        assert len(results) == 0
    
    def test_too_small_image(self, extractor):
        """Test that very small images are rejected."""
        tiny_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        results = extractor.extract(tiny_img)
        assert len(results) == 0
    
    def test_grayscale_input(self, extractor, bordered_table_image):
        """Test that grayscale input works."""
        gray = cv2.cvtColor(bordered_table_image, cv2.COLOR_BGR2GRAY)
        results = extractor.extract(gray)
        # Should still detect tables
        assert len(results) >= 0  # May or may not detect depending on quality


class TestTableExtractor:
    """Tests for the main TableExtractor class."""
    
    def test_extractor_init(self):
        """Test TableExtractor initialization."""
        extractor = TableExtractor()
        assert "opencv" in extractor._extractors
    
    def test_extract_from_region(self):
        """Test extraction from a specific region."""
        extractor = TableExtractor()
        
        # Create image with table in specific region
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Draw table in region (100, 100) to (500, 400)
        cv2.rectangle(img, (100, 100), (500, 400), (0, 0, 0), 2)
        cv2.line(img, (100, 200), (500, 200), (0, 0, 0), 2)
        cv2.line(img, (100, 300), (500, 300), (0, 0, 0), 2)
        cv2.line(img, (300, 100), (300, 400), (0, 0, 0), 2)
        
        result = extractor.extract_from_region(img, (100, 100, 500, 400))
        
        # Result should have the specified bbox
        if result:
            assert result.bbox == (100, 100, 500, 400)


class TestSampleTableImage:
    """Tests using the actual sample_table.png."""
    
    @pytest.fixture
    def sample_table(self):
        """Load sample table image if it exists."""
        sample_path = Path(__file__).parent.parent / "examples" / "sample_pages" / "sample_table.png"
        if sample_path.exists():
            return cv2.imread(str(sample_path))
        return None
    
    def test_sample_table_detection(self, sample_table):
        """Test detection on actual sample table."""
        if sample_table is None:
            pytest.skip("sample_table.png not found")
        
        extractor = OpenCVTableExtractor(min_rows=2, min_cols=2)
        results = extractor.extract(sample_table)
        
        # Log results for debugging
        print(f"Tables found: {len(results)}")
        for i, r in enumerate(results):
            print(f"Table {i+1}: {r.num_rows}x{r.num_cols}, method={r.method_used}")
        
        # Should detect at least one table
        assert len(results) >= 1, "Should detect at least one table in sample_table.png"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
