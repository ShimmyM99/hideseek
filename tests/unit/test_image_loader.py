import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from hideseek.core.image_loader import HideSeekImageLoader


class TestHideSeekImageLoader:
    """Test suite for HideSeekImageLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create ImageLoader instance for testing"""
        return HideSeekImageLoader()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :50] = [255, 0, 0]  # Red square
        img[50:, 50:] = [0, 255, 0]  # Green square
        img[:50, 50:] = [0, 0, 255]  # Blue square
        return img
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create temporary image file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, sample_image)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_init(self, loader):
        """Test ImageLoader initialization"""
        assert loader is not None
        assert hasattr(loader, 'supported_formats')
        assert hasattr(loader, 'max_resolution')
        assert '.jpg' in loader.supported_formats
        assert '.png' in loader.supported_formats
    
    def test_load_test_image_success(self, loader, temp_image_file):
        """Test successful image loading"""
        img = loader.load_test_image(temp_image_file)
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Color image
        assert img.shape[2] == 3    # BGR channels
        assert img.dtype == np.uint8
    
    def test_load_test_image_file_not_found(self, loader):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            loader.load_test_image('nonexistent_file.jpg')
    
    def test_load_test_image_unsupported_format(self, loader):
        """Test loading unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            try:
                with pytest.raises(ValueError, match="Unsupported image format"):
                    loader.load_test_image(tmp.name)
            finally:
                os.unlink(tmp.name)
    
    @patch('cv2.imread')
    def test_load_test_image_corrupted(self, mock_imread, loader, temp_image_file):
        """Test loading corrupted image"""
        mock_imread.return_value = None
        with pytest.raises(ValueError, match="Could not load image"):
            loader.load_test_image(temp_image_file)
    
    def test_validate_image_format_valid(self, loader, sample_image):
        """Test validation of valid image"""
        assert loader.validate_image_format(sample_image) == True
    
    def test_validate_image_format_none(self, loader):
        """Test validation of None image"""
        assert loader.validate_image_format(None) == False
    
    def test_validate_image_format_wrong_type(self, loader):
        """Test validation of wrong data type"""
        assert loader.validate_image_format("not an array") == False
    
    def test_validate_image_format_wrong_dimensions(self, loader):
        """Test validation of wrong dimensions"""
        # 1D array
        assert loader.validate_image_format(np.array([1, 2, 3])) == False
        # 4D array  
        assert loader.validate_image_format(np.zeros((10, 10, 3, 2))) == False
    
    def test_validate_image_format_wrong_channels(self, loader):
        """Test validation of wrong number of channels"""
        # 5 channels
        assert loader.validate_image_format(np.zeros((10, 10, 5))) == False
    
    def test_validate_image_format_too_small(self, loader):
        """Test validation of too small image"""
        small_img = np.zeros((5, 5, 3), dtype=np.uint8)
        assert loader.validate_image_format(small_img) == False
    
    def test_get_image_info(self, loader, sample_image):
        """Test getting image information"""
        info = loader.get_image_info(sample_image)
        
        assert 'shape' in info
        assert 'dtype' in info
        assert 'channels' in info
        assert 'width' in info
        assert 'height' in info
        assert 'min_value' in info
        assert 'max_value' in info
        assert 'mean_value' in info
        assert 'std_value' in info
        
        assert info['shape'] == sample_image.shape
        assert info['channels'] == 3
        assert info['width'] == 100
        assert info['height'] == 100
    
    def test_preprocess_image(self, loader, sample_image):
        """Test image preprocessing"""
        processed = loader.preprocess_image(sample_image)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == sample_image.shape
    
    def test_preprocess_image_resize_large(self, loader):
        """Test resizing large image"""
        # Create oversized image
        large_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        processed = loader.preprocess_image(large_img)
        
        # Should be resized to within max_resolution
        assert processed.shape[0] <= loader.max_resolution[1]
        assert processed.shape[1] <= loader.max_resolution[0]
    
    def test_load_batch_images_empty_directory(self, loader):
        """Test loading from empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            images = loader.load_batch_images(temp_dir)
            assert len(images) == 0
    
    def test_load_batch_images_directory_not_found(self, loader):
        """Test loading from non-existent directory"""
        with pytest.raises(FileNotFoundError):
            loader.load_batch_images('nonexistent_directory')
    
    def test_load_batch_images_not_directory(self, loader, temp_image_file):
        """Test loading when path is not a directory"""
        with pytest.raises(ValueError, match="Path is not a directory"):
            loader.load_batch_images(temp_image_file)
    
    def test_load_batch_images_success(self, loader, sample_image):
        """Test successful batch loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test images
            for i in range(3):
                img_path = Path(temp_dir) / f'test_{i}.jpg'
                cv2.imwrite(str(img_path), sample_image)
            
            images = loader.load_batch_images(temp_dir)
            assert len(images) == 3
            
            for img_data in images:
                assert 'path' in img_data
                assert 'filename' in img_data
                assert 'image' in img_data
                assert isinstance(img_data['image'], np.ndarray)
    
    def test_extract_metadata(self, loader, temp_image_file):
        """Test metadata extraction"""
        metadata = loader.extract_metadata(temp_image_file)
        
        assert 'filename' in metadata
        assert 'file_size' in metadata
        assert 'format' in metadata
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'mode' in metadata
        
        assert metadata['format'] == '.jpg'
        assert metadata['file_size'] > 0
        assert isinstance(metadata['width'], int)
        assert isinstance(metadata['height'], int)
    
    def test_extract_metadata_nonexistent_file(self, loader):
        """Test metadata extraction from non-existent file"""
        metadata = loader.extract_metadata('nonexistent.jpg')
        # Should not raise exception, just return basic metadata
        assert 'filename' in metadata
        assert 'format' in metadata


if __name__ == '__main__':
    pytest.main([__file__])