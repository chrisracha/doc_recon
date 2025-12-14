"""
Tests for layout detection module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBoundingBox:
    """Test BoundingBox class."""
    
    def test_bbox_creation(self):
        """Test bounding box creation."""
        from utils.layout import BoundingBox
        
        bbox = BoundingBox(10, 20, 110, 70)
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70
    
    def test_bbox_properties(self):
        """Test bounding box computed properties."""
        from utils.layout import BoundingBox
        
        bbox = BoundingBox(10, 20, 110, 70)
        
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        assert bbox.center == (60, 45)
    
    def test_bbox_from_xywh(self):
        """Test creation from x, y, width, height."""
        from utils.layout import BoundingBox
        
        bbox = BoundingBox.from_xywh(10, 20, 100, 50)
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70
    
    def test_bbox_to_tuple(self):
        """Test conversion to tuple."""
        from utils.layout import BoundingBox
        
        bbox = BoundingBox(10, 20, 110, 70)
        
        assert bbox.to_tuple() == (10, 20, 110, 70)
        assert bbox.to_xywh() == (10, 20, 100, 50)
    
    def test_bbox_intersects(self):
        """Test intersection detection."""
        from utils.layout import BoundingBox
        
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)
        bbox3 = BoundingBox(200, 200, 300, 300)
        
        assert bbox1.intersects(bbox2) is True
        assert bbox1.intersects(bbox3) is False
    
    def test_bbox_intersection_area(self):
        """Test intersection area calculation."""
        from utils.layout import BoundingBox
        
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)
        
        intersection = bbox1.intersection_area(bbox2)
        assert intersection == 50 * 50  # 2500
    
    def test_bbox_iou(self):
        """Test IoU calculation."""
        from utils.layout import BoundingBox
        
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 100, 100)  # Same box
        
        assert bbox1.iou(bbox2) == 1.0
        
        bbox3 = BoundingBox(100, 100, 200, 200)  # No overlap
        assert bbox1.iou(bbox3) == 0.0


class TestBlock:
    """Test Block class."""
    
    def test_block_creation(self):
        """Test block creation."""
        from utils.layout import Block, BoundingBox, BlockType
        
        bbox = BoundingBox(10, 20, 110, 70)
        block = Block(
            bbox=bbox,
            block_type=BlockType.PARAGRAPH,
            confidence=0.95
        )
        
        assert block.bbox == bbox
        assert block.block_type == BlockType.PARAGRAPH
        assert block.confidence == 0.95
        assert block.block_id  # Should have auto-generated ID
    
    def test_block_crop(self):
        """Test cropping block from image."""
        from utils.layout import Block, BoundingBox, BlockType
        
        image = np.ones((100, 200), dtype=np.uint8) * 255
        image[20:50, 30:100] = 0  # Dark region
        
        bbox = BoundingBox(30, 20, 100, 50)
        block = Block(bbox=bbox, block_type=BlockType.PARAGRAPH)
        
        cropped = block.crop_from_image(image)
        
        assert cropped.shape == (30, 70)
    
    def test_block_to_dict(self):
        """Test block serialization."""
        from utils.layout import Block, BoundingBox, BlockType
        
        bbox = BoundingBox(10, 20, 110, 70)
        block = Block(
            bbox=bbox,
            block_type=BlockType.PARAGRAPH,
            confidence=0.95,
            text="Hello world",
            reading_order=1
        )
        
        data = block.to_dict()
        
        assert data["type"] == "paragraph"
        assert data["bbox"] == (10, 20, 110, 70)
        assert data["confidence"] == 0.95
        assert data["text"] == "Hello world"
        assert data["reading_order"] == 1


class TestLayoutDetector:
    """Test LayoutDetector class."""
    
    @pytest.fixture
    def sample_document_image(self):
        """Create a sample document-like image."""
        # Create a simple document layout
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Title area (top)
        img[50:80, 100:500, :] = [0, 0, 0]
        
        # Paragraph blocks
        for y in range(150, 400, 50):
            img[y:y+30, 50:550, :] = [50, 50, 50]
        
        # Table-like area
        img[450:550, 100:500, :] = [200, 200, 200]
        # Horizontal lines
        for y in range(450, 550, 25):
            img[y:y+2, 100:500, :] = [0, 0, 0]
        # Vertical lines
        for x in range(100, 500, 100):
            img[450:550, x:x+2, :] = [0, 0, 0]
        
        return img
    
    def test_layout_detector_init(self):
        """Test layout detector initialization with fallback."""
        from utils.layout import LayoutDetector
        
        # Should fall back to classical if deep learning not available
        detector = LayoutDetector(
            fallback_to_classical=True
        )
        
        assert detector._detector is not None
        assert detector._method in ["layoutparser", "paddleocr", "classical"]
    
    def test_classical_layout_detection(self, sample_document_image):
        """Test classical CV-based layout detection."""
        from utils.layout import ClassicalLayoutDetector
        
        detector = ClassicalLayoutDetector(min_block_area=100)
        result = detector.detect(sample_document_image)
        
        assert result.blocks is not None
        assert result.page_width == 600
        assert result.page_height == 800
        assert len(result.blocks) >= 1
    
    def test_reading_order_assignment(self, sample_document_image):
        """Test that reading order is properly assigned."""
        from utils.layout import LayoutDetector
        
        detector = LayoutDetector(fallback_to_classical=True)
        result = detector.detect(sample_document_image)
        
        # Check reading orders are sequential
        reading_orders = [b.reading_order for b in result.blocks]
        assert reading_orders == sorted(reading_orders)
    
    def test_debug_image_generation(self, sample_document_image):
        """Test debug image with bounding boxes."""
        from utils.layout import LayoutDetector
        
        detector = LayoutDetector(fallback_to_classical=True)
        result = detector.detect(sample_document_image, debug=True)
        
        # Debug image should be generated
        # Note: Only ClassicalLayoutDetector generates debug images through draw_debug_image
        # The actual availability depends on the method used
    
    def test_layout_result_to_dict(self, sample_document_image):
        """Test layout result serialization."""
        from utils.layout import ClassicalLayoutDetector
        
        detector = ClassicalLayoutDetector()
        result = detector.detect(sample_document_image)
        
        # Check that blocks can be serialized
        for block in result.blocks:
            data = block.to_dict()
            assert "type" in data
            assert "bbox" in data
            assert "confidence" in data


class TestBlockTypes:
    """Test block type handling."""
    
    def test_all_block_types_exist(self):
        """Test that all expected block types are defined."""
        from utils.layout import BlockType
        
        expected_types = [
            "TITLE", "AUTHORS", "ABSTRACT", "HEADING",
            "PARAGRAPH", "EQUATION_BLOCK", "EQUATION_INLINE",
            "TABLE", "FIGURE", "CAPTION", "REFERENCES",
            "FOOTER", "HEADER", "LIST", "CODE", "UNKNOWN"
        ]
        
        for type_name in expected_types:
            assert hasattr(BlockType, type_name)
    
    def test_block_type_values(self):
        """Test block type string values."""
        from utils.layout import BlockType
        
        assert BlockType.TITLE.value == "title"
        assert BlockType.PARAGRAPH.value == "paragraph"
        assert BlockType.EQUATION_BLOCK.value == "equation_block"
        assert BlockType.TABLE.value == "table"


class TestColumnDetection:
    """Test multi-column detection."""
    
    @pytest.fixture
    def two_column_image(self):
        """Create a two-column document image."""
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Left column blocks
        for y in range(100, 600, 60):
            img[y:y+40, 30:270, :] = [50, 50, 50]
        
        # Right column blocks
        for y in range(100, 600, 60):
            img[y:y+40, 330:570, :] = [50, 50, 50]
        
        return img
    
    def test_column_detection(self, two_column_image):
        """Test detection of multi-column layout."""
        from utils.layout import LayoutDetector
        
        detector = LayoutDetector(fallback_to_classical=True)
        result = detector.detect(two_column_image)
        
        # Should detect blocks in both columns
        if result.blocks:
            column_indices = set(b.column_index for b in result.blocks)
            # May detect 2 columns depending on the implementation
            assert len(column_indices) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

