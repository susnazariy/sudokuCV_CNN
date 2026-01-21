"""Image preprocessing utilities for Sudoku detection."""

import cv2
import numpy as np


def calculate_block_size(image: np.ndarray, scale_factor: float = 0.02, 
                         min_size: int = 3, max_size: int = 51) -> int:
    """Calculate adaptive block size for thresholding based on image dimensions."""
    h, w = image.shape[:2]
    smaller_dim = min(h, w)
    
    block_size = int(smaller_dim * scale_factor)
    block_size = max(min_size, min(block_size, max_size))
    
    # Must be odd
    if block_size % 2 == 0:
        block_size += 1
    
    return block_size


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to binary threshold for contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    block_size = calculate_block_size(blurred)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, 2
    )
    return thresh
