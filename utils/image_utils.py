#!/usr/bin/env python3
"""
Image Processing Utilities for Elephant Identification System
Common image processing functions used across the application
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224),
                             normalize: bool = True) -> np.ndarray:
    """
    Load and preprocess image for model input

    Args:
        image_path: Path to image file
        target_size: Target dimensions (width, height)
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize if requested
        if normalize:
            image = image.astype(np.float32) / 255.0

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise

def get_image_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Get all image files from a directory

    Args:
        directory: Directory path to search
        recursive: Whether to search subdirectories

    Returns:
        List of image file paths
    """
    image_files = []

    try:
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []

        # Search pattern
        pattern = '**/*' if recursive else '*'

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                image_files.append(str(file_path))

        logger.info(f"Found {len(image_files)} image files in {directory}")
        return sorted(image_files)

    except Exception as e:
        logger.error(f"Error getting image files from {directory}: {e}")
        return []

def validate_image(image_path: str) -> bool:
    """
    Validate if file is a valid image

    Args:
        image_path: Path to image file

    Returns:
        True if valid image, False otherwise
    """
    try:
        # Check file extension
        if not Path(image_path).suffix.lower() in SUPPORTED_FORMATS:
            return False

        # Try to load with PIL
        with Image.open(image_path) as img:
            img.verify()

        # Try to load with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return False

        return True

    except Exception as e:
        logger.debug(f"Image validation failed for {image_path}: {e}")
        return False

def get_image_info(image_path: str) -> dict:
    """
    Get detailed information about an image

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image information
    """
    try:
        # Basic file info
        file_path = Path(image_path)
        file_size = file_path.stat().st_size

        # Image info using PIL
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format

        # Additional info using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            channels = image.shape[2] if len(image.shape) == 3 else 1
        else:
            channels = len(mode) if mode else 1

        info = {
            'filename': file_path.name,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'dimensions': (width, height),
            'width': width,
            'height': height,
            'channels': channels,
            'mode': mode,
            'format': format_name,
            'aspect_ratio': round(width / height, 2) if height > 0 else 0,
            'megapixels': round((width * height) / 1_000_000, 2)
        }

        return info

    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return {'error': str(e)}

def enhance_image(image: np.ndarray, brightness: float = 1.0,
                 contrast: float = 1.0, sharpness: float = 1.0) -> np.ndarray:
    """
    Enhance image quality

    Args:
        image: Input image as numpy array
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        sharpness: Sharpness factor (1.0 = no change)

    Returns:
        Enhanced image
    """
    try:
        # Convert to PIL for enhancement
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)

        # Convert back to numpy
        enhanced_image = np.array(pil_image)

        return enhanced_image

    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image

def crop_to_square(image: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """
    Crop image to square shape

    Args:
        image: Input image
        size: Target size (if None, uses minimum dimension)

    Returns:
        Square cropped image
    """
    try:
        height, width = image.shape[:2]

        if size is None:
            size = min(height, width)

        # Calculate crop coordinates
        center_x, center_y = width // 2, height // 2
        half_size = size // 2

        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, center_x + half_size)
        y2 = min(height, center_y + half_size)

        # Crop image
        cropped = image[y1:y2, x1:x2]

        # Resize to exact square if needed
        if cropped.shape[0] != size or cropped.shape[1] != size:
            cropped = cv2.resize(cropped, (size, size))

        return cropped

    except Exception as e:
        logger.error(f"Error cropping to square: {e}")
        return image

def detect_ears(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Simple ear detection using contour analysis
    This is a basic implementation - replace with trained model for better results

    Args:
        image: Input image

    Returns:
        List of bounding boxes (x, y, w, h) for detected ears
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours that might be ears
        ear_boxes = []
        height, width = gray.shape
        min_area = (height * width) * 0.01  # Minimum 1% of image area
        max_area = (height * width) * 0.3   # Maximum 30% of image area

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Basic shape filtering (ears tend to be taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 3.0:  # Reasonable aspect ratio for ears
                    ear_boxes.append((x, y, w, h))

        # Sort by area (largest first)
        ear_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)

        # Return top 2 candidates (for two ears)
        return ear_boxes[:2]

    except Exception as e:
        logger.error(f"Error detecting ears: {e}")
        return []

def create_pytorch_transforms(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    """
    Create PyTorch transforms for image preprocessing

    Args:
        image_size: Target image size
        augment: Whether to include data augmentation

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]

    if augment:
        # Insert augmentation transforms before normalization
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ]

        # Insert augmentations before ToTensor
        transform_list = transform_list[:1] + augment_transforms + transform_list[1:]

    return transforms.Compose(transform_list)

def batch_process_images(image_paths: List[str], process_func, batch_size: int = 32,
                        **kwargs) -> List:
    """
    Process images in batches

    Args:
        image_paths: List of image paths
        process_func: Function to apply to each image
        batch_size: Number of images to process at once
        **kwargs: Additional arguments for process_func

    Returns:
        List of processing results
    """
    results = []

    try:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            batch_results = []
            for image_path in batch_paths:
                try:
                    result = process_func(image_path, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    batch_results.append(None)

            results.extend(batch_results)

            # Log progress
            processed = min(i + batch_size, len(image_paths))
            logger.info(f"Processed {processed}/{len(image_paths)} images")

        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return results

def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    Save image to file

    Args:
        image: Image array to save
        output_path: Output file path
        quality: JPEG quality (if applicable)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to proper format for saving
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Save using OpenCV (expects BGR)
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Set quality for JPEG
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            success = cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success = cv2.imwrite(output_path, image_bgr)

        if success:
            logger.debug(f"Saved image to {output_path}")
            return True
        else:
            logger.error(f"Failed to save image to {output_path}")
            return False

    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False

def calculate_image_statistics(image_paths: List[str]) -> dict:
    """
    Calculate statistics across multiple images

    Args:
        image_paths: List of image paths to analyze

    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'total_size_mb': 0,
            'dimensions': [],
            'formats': {},
            'total_pixels': 0
        }

        for image_path in image_paths:
            try:
                info = get_image_info(image_path)

                if 'error' not in info:
                    stats['valid_images'] += 1
                    stats['total_size_mb'] += info['file_size_mb']
                    stats['dimensions'].append((info['width'], info['height']))
                    stats['total_pixels'] += info['width'] * info['height']

                    # Count formats
                    format_name = info.get('format', 'Unknown')
                    stats['formats'][format_name] = stats['formats'].get(format_name, 0) + 1

            except Exception as e:
                logger.debug(f"Error analyzing {image_path}: {e}")
                continue

        # Calculate summary statistics
        if stats['dimensions']:
            widths = [d[0] for d in stats['dimensions']]
            heights = [d[1] for d in stats['dimensions']]

            stats['average_width'] = np.mean(widths)
            stats['average_height'] = np.mean(heights)
            stats['min_width'] = np.min(widths)
            stats['max_width'] = np.max(widths)
            stats['min_height'] = np.min(heights)
            stats['max_height'] = np.max(heights)
            stats['average_megapixels'] = stats['total_pixels'] / (stats['valid_images'] * 1_000_000)

        return stats

    except Exception as e:
        logger.error(f"Error calculating image statistics: {e}")
        return {'error': str(e)}
