"""
Tests for image preprocessing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPreprocessing:
    """Test image preprocessing functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample grayscale image."""
        # Create a simple test image with some text-like patterns
        img = np.ones((300, 400), dtype=np.uint8) * 255
        # Add some dark regions (simulating text)
        img[50:60, 50:200] = 0
        img[80:90, 50:180] = 0
        img[110:120, 50:220] = 0
        return img
    
    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image."""
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        img[50:60, 50:200] = [0, 0, 0]
        img[80:90, 50:180] = [0, 0, 0]
        return img
    
    @pytest.fixture
    def skewed_image(self):
        """Create a slightly skewed image."""
        import cv2
        
        # Create base image
        img = np.ones((400, 500), dtype=np.uint8) * 255
        
        # Add horizontal lines
        for y in range(50, 350, 30):
            img[y:y+2, 50:450] = 0
        
        # Rotate slightly
        center = (250, 200)
        angle = 5.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rotation_matrix, (500, 400),
                                  borderValue=255)
        
        return rotated
    
    def test_to_grayscale_already_gray(self, sample_image):
        """Test that grayscale images are returned unchanged."""
        from utils.images import to_grayscale
        
        result = to_grayscale(sample_image)
        
        assert result.shape == sample_image.shape
        assert len(result.shape) == 2
        np.testing.assert_array_equal(result, sample_image)
    
    def test_to_grayscale_from_color(self, sample_color_image):
        """Test conversion from color to grayscale."""
        from utils.images import to_grayscale
        
        result = to_grayscale(sample_color_image)
        
        assert len(result.shape) == 2
        assert result.shape[:2] == sample_color_image.shape[:2]
    
    def test_deskew_straight_image(self, sample_image):
        """Test that straight images are not significantly changed."""
        from utils.images import deskew
        
        result, angle = deskew(sample_image, return_angle=True)
        
        # Should detect minimal skew
        assert abs(angle) < 2.0
    
    def test_deskew_skewed_image(self, skewed_image):
        """Test deskewing of a rotated image."""
        from utils.images import deskew
        
        result, angle = deskew(skewed_image, max_angle=15.0, return_angle=True)
        
        # Should detect the 5 degree rotation
        assert abs(angle - 5.0) < 3.0  # Allow some tolerance
    
    def test_denoise(self, sample_image):
        """Test denoising function."""
        from utils.images import denoise
        
        # Add some noise
        noisy = sample_image.copy()
        noise = np.random.normal(0, 25, sample_image.shape).astype(np.uint8)
        noisy = np.clip(noisy.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        
        result = denoise(noisy, strength=10)
        
        assert result.shape == noisy.shape
        # Denoised image should have less variation
        assert np.std(result) <= np.std(noisy)
    
    def test_binarize_otsu(self, sample_image):
        """Test Otsu binarization."""
        from utils.images import binarize
        
        result = binarize(sample_image, method="otsu")
        
        assert result.shape == sample_image.shape
        # Should only have 0 and 255 values
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)
    
    def test_binarize_adaptive(self, sample_image):
        """Test adaptive binarization."""
        from utils.images import binarize
        
        result = binarize(sample_image, method="adaptive")
        
        assert result.shape == sample_image.shape
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_enhance_contrast(self, sample_image):
        """Test contrast enhancement."""
        from utils.images import enhance_contrast
        
        # Create a low-contrast image
        low_contrast = (sample_image * 0.5 + 64).astype(np.uint8)
        
        result = enhance_contrast(low_contrast, clip_limit=2.0)
        
        assert result.shape == low_contrast.shape
        # Enhanced image should have greater dynamic range
        assert np.std(result) >= np.std(low_contrast)
    
    def test_enhance_contrast_color(self, sample_color_image):
        """Test contrast enhancement on color images."""
        from utils.images import enhance_contrast
        
        result = enhance_contrast(sample_color_image)
        
        assert result.shape == sample_color_image.shape
        assert len(result.shape) == 3
    
    def test_preprocess_pipeline(self, sample_image):
        """Test the full preprocessing pipeline."""
        from utils.images import preprocess_image
        
        result = preprocess_image(
            sample_image,
            deskew_enabled=True,
            denoise_enabled=True,
            enhance_contrast_enabled=True,
            binarize_enabled=False
        )
        
        assert result.image is not None
        assert result.original_shape == sample_image.shape[:2]
        assert isinstance(result.transformations, list)
        assert len(result.transformations) > 0
    
    def test_preprocess_pipeline_with_binarize(self, sample_image):
        """Test preprocessing pipeline with binarization."""
        from utils.images import preprocess_image
        
        result = preprocess_image(
            sample_image,
            deskew_enabled=False,
            denoise_enabled=False,
            enhance_contrast_enabled=False,
            binarize_enabled=True
        )
        
        # Should be binary
        unique_values = np.unique(result.image)
        assert len(unique_values) <= 2
        assert result.was_binarized is True
    
    def test_get_image_stats(self, sample_image):
        """Test image statistics calculation."""
        from utils.images import get_image_stats
        
        stats = get_image_stats(sample_image)
        
        assert stats.height == 300
        assert stats.width == 400
        assert stats.channels == 1
        assert 0 <= stats.mean_intensity <= 255
        assert stats.std_intensity >= 0
        assert stats.is_grayscale is True
    
    def test_draw_debug_image(self, sample_image):
        """Test debug image drawing."""
        from utils.images import draw_debug_image
        
        boxes = [(50, 50, 150, 10), (50, 80, 130, 10)]
        labels = ["line1", "line2"]
        
        result = draw_debug_image(sample_image, boxes, labels)
        
        assert result.shape[0] == sample_image.shape[0]
        assert result.shape[1] == sample_image.shape[1]
        assert len(result.shape) == 3  # Should be color
    
    def test_create_comparison_image(self, sample_image):
        """Test comparison image creation."""
        from utils.images import create_comparison_image
        
        processed = sample_image.copy()
        processed[processed > 128] = 255
        processed[processed <= 128] = 0
        
        result = create_comparison_image(sample_image, processed)
        
        # Should be wider than original (side by side + separator)
        assert result.shape[1] > sample_image.shape[1]
        assert len(result.shape) == 3


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_image(self):
        """Test handling of empty images."""
        from utils.images import preprocess_image
        
        # Very small image
        tiny = np.ones((2, 2), dtype=np.uint8) * 128
        
        result = preprocess_image(tiny, deskew_enabled=False)
        
        assert result.image is not None
    
    def test_single_channel_4d(self):
        """Test handling of images with extra dimensions."""
        from utils.images import to_grayscale
        
        # 4-channel image (BGRA)
        img = np.ones((100, 100, 4), dtype=np.uint8) * 128
        
        result = to_grayscale(img)
        
        assert len(result.shape) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

