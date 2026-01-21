"""Grid detection and perspective transformation utilities."""

import cv2
import numpy as np
from typing import Optional, Tuple


def add_border(image: np.ndarray, border_size: int = 10, color: int = 255) -> np.ndarray:
    """Add border/padding around image."""
    return cv2.copyMakeBorder(
        image, 
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, 
        value=color
    )


def _find_quadrilateral(thresh: np.ndarray, min_area_ratio: float = 0.05) -> Optional[np.ndarray]:
    """Find the largest quadrilateral contour in a thresholded image."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    image_area = thresh.shape[0] * thresh.shape[1]
    min_area = image_area * min_area_ratio
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        
        if len(approx) == 4:
            return approx
    
    return None


def find_grid_contour(thresh: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Find the largest quadrilateral contour (the Sudoku grid).
    
    Strategy:
        1. Try to find contour normally
        2. If not found, add white padding and try again
    """
    # First attempt: find contour without padding
    contour = _find_quadrilateral(thresh)
    if contour is not None:
        return contour
    
    # Second attempt: add white padding and try again
    border_size = min(3, min(thresh.shape) // 20)
    padded = add_border(thresh, border_size, color=255)  # White padding
    
    contour = _find_quadrilateral(padded)
    if contour is not None:
        # Adjust coordinates back (subtract padding offset)
        contour = contour - border_size
        return contour
    
    return None


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    corners = corners.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).flatten()
    
    ordered[0] = corners[np.argmin(s)]  # Top-left
    ordered[1] = corners[np.argmin(d)]  # Top-right
    ordered[2] = corners[np.argmax(s)]  # Bottom-right
    ordered[3] = corners[np.argmax(d)]  # Bottom-left
    
    return ordered


def warp_perspective(image: np.ndarray, corners: np.ndarray, 
                     output_size: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the grid region to a square image."""
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))
    
    return warped, matrix
