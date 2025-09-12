import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from PIL import Image, ExifTags
import os

from ..config import config
from ..utils.logging_config import get_logger

logger = get_logger('image_loader')


class HideSeekImageLoader:
    """
    Professional image loader for HideSeek camouflage analysis system.
    Handles loading, validation, preprocessing, and metadata extraction.
    """
    
    def __init__(self):
        self.supported_formats = config.get('image.supported_formats', 
                                           ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        self.max_resolution = config.get('image.max_resolution', [1920, 1080])
        self.preprocessing_enabled = config.get('image.preprocessing', {})
        
        logger.info(f"ImageLoader initialized with formats: {self.supported_formats}")
        logger.info(f"Max resolution: {self.max_resolution[0]}x{self.max_resolution[1]}")
    
    def load_test_image(self, path: str) -> np.ndarray:
        """
        Load a single test image with validation and preprocessing.
        
        Args:
            path: Path to the image file
            
        Returns:
            Loaded and preprocessed image as numpy array (BGR format)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format not supported or corrupted
        """
        image_path = Path(path)
        
        # Validate file existence
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Validate file format
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        logger.debug(f"Loading image: {path}")
        
        try:
            # Load image using OpenCV (BGR format)
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Could not load image (corrupted?): {path}")
            
            # Validate image
            if not self.validate_image_format(img):
                raise ValueError(f"Invalid image data: {path}")
            
            # Apply preprocessing
            img = self.preprocess_image(img)
            
            logger.debug(f"Successfully loaded image: {img.shape}")
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image {path}: {str(e)}")
            raise
    
    def load_batch_images(self, directory: str, pattern: str = "*") -> List[Dict[str, Union[str, np.ndarray]]]:
        """
        Load multiple images from a directory.
        
        Args:
            directory: Directory path containing images
            pattern: File pattern to match (default: all files)
            
        Returns:
            List of dictionaries containing 'path', 'filename', and 'image' data
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        logger.info(f"Loading batch images from: {directory}")
        
        loaded_images = []
        
        # Find all image files matching the pattern
        for ext in self.supported_formats:
            search_pattern = f"{pattern}{ext}" if pattern != "*" else f"*{ext}"
            for image_path in dir_path.glob(search_pattern):
                try:
                    img = self.load_test_image(str(image_path))
                    loaded_images.append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'image': img
                    })
                    logger.debug(f"Loaded: {image_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {image_path.name}: {str(e)}")
                    continue
        
        logger.info(f"Successfully loaded {len(loaded_images)} images")
        return loaded_images
    
    def validate_image_format(self, img: np.ndarray) -> bool:
        """
        Validate loaded image format and properties.
        
        Args:
            img: Image array to validate
            
        Returns:
            True if image is valid, False otherwise
        """
        if img is None:
            return False
        
        # Check if it's a valid numpy array
        if not isinstance(img, np.ndarray):
            return False
        
        # Check dimensions (should be 2D or 3D)
        if len(img.shape) not in [2, 3]:
            return False
        
        # Check for color images (3 channels)
        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            return False
        
        # Check data type
        if img.dtype not in [np.uint8, np.uint16, np.float32]:
            return False
        
        # Check for reasonable dimensions
        height, width = img.shape[:2]
        if height < 10 or width < 10:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        if height > self.max_resolution[1] or width > self.max_resolution[0]:
            logger.info(f"Image exceeds max resolution: {width}x{height}")
            # Don't fail validation, just log - we'll resize in preprocessing
        
        return True
    
    def extract_metadata(self, path: str) -> Dict[str, any]:
        """
        Extract metadata from image file including EXIF data.
        
        Args:
            path: Path to image file
            
        Returns:
            Dictionary containing metadata
        """
        image_path = Path(path)
        metadata = {
            'filename': image_path.name,
            'file_size': image_path.stat().st_size,
            'format': image_path.suffix.lower(),
            'exif': {}
        }
        
        try:
            # Get basic image info
            with Image.open(image_path) as pil_img:
                metadata.update({
                    'width': pil_img.width,
                    'height': pil_img.height,
                    'mode': pil_img.mode,
                    'has_transparency': pil_img.mode in ('RGBA', 'LA', 'P')
                })
                
                # Extract EXIF data
                if hasattr(pil_img, '_getexif'):
                    exif_data = pil_img._getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            metadata['exif'][tag] = value
        
        except Exception as e:
            logger.warning(f"Could not extract metadata from {path}: {str(e)}")
        
        return metadata
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to the loaded image.
        
        Args:
            img: Input image array
            
        Returns:
            Preprocessed image array
        """
        processed_img = img.copy()
        
        try:
            # Resize if exceeds max resolution
            height, width = processed_img.shape[:2]
            if height > self.max_resolution[1] or width > self.max_resolution[0]:
                # Calculate scaling factor maintaining aspect ratio
                scale_h = self.max_resolution[1] / height
                scale_w = self.max_resolution[0] / width
                scale = min(scale_h, scale_w)
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                processed_img = cv2.resize(processed_img, (new_width, new_height), 
                                         interpolation=cv2.INTER_LANCZOS4)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Apply denoising if enabled
            if self.preprocessing_enabled.get('denoise', False):
                processed_img = cv2.fastNlMeansDenoisingColored(processed_img, None, 10, 10, 7, 21)
                logger.debug("Applied denoising")
            
            # Enhance contrast if enabled
            if self.preprocessing_enabled.get('enhance_contrast', False):
                lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processed_img = cv2.merge([l, a, b])
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)
                logger.debug("Applied contrast enhancement")
            
            # Normalize if enabled
            if self.preprocessing_enabled.get('normalize', False):
                processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
                logger.debug("Applied normalization")
            
            return processed_img
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return img  # Return original image if preprocessing fails
    
    def get_image_info(self, img: np.ndarray) -> Dict[str, any]:
        """
        Get information about a loaded image array.
        
        Args:
            img: Image array
            
        Returns:
            Dictionary containing image information
        """
        return {
            'shape': img.shape,
            'dtype': str(img.dtype),
            'channels': img.shape[2] if len(img.shape) == 3 else 1,
            'width': img.shape[1],
            'height': img.shape[0],
            'min_value': np.min(img),
            'max_value': np.max(img),
            'mean_value': np.mean(img),
            'std_value': np.std(img)
        }