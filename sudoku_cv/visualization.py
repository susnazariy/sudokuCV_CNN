"""Visualization utilities for Sudoku solving."""

import cv2
import numpy as np
from typing import Tuple


def draw_corners(image: np.ndarray, corners: np.ndarray, 
                 contour: np.ndarray = None) -> np.ndarray:
    """Draw detected corners on the image."""
    vis = image.copy()
    h, w = image.shape[:2]
    pts = corners.reshape(-1, 2).astype(int)
    
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
    else:
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    
    offsets = [
        (20, 20),    # TL
        (-20, 20),   # TR
        (-20, -20),  # BR
        (20, -20)    # BL
    ]
    labels = ['0', '1', '2', '3']
    
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 7, (0, 0, 255), -1)
        
        ox, oy = offsets[i]
        tx = max(5, min(x + ox, w - 40))
        ty = max(20, min(y + oy, h - 5))
        
        cv2.putText(vis, labels[i], (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis


def visualize_grid_lines(warped: np.ndarray, h_lines: list, v_lines: list) -> np.ndarray:
    """Visualize detected grid lines."""
    vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if len(warped.shape) == 2 else warped.copy()
    
    for y in h_lines:
        cv2.line(vis, (0, y), (vis.shape[1], y), (255, 0, 0), 1)
    
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 255, 0), 1)
    
    return vis


def estimate_font_scale(corners: np.ndarray) -> float:
    """Estimate appropriate font scale based on grid size."""
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])
    cell_size = min(width, height) / 9
    
    font_scale = cell_size / 50
    return max(0.5, min(font_scale, 3.0))


def draw_solution_on_image(original_image: np.ndarray, corners: np.ndarray,
                           original_puzzle: np.ndarray, solution: np.ndarray,
                           grid_size: int = 450, color: Tuple[int, int, int] = (0, 180, 0),
                           thickness: int = 2) -> np.ndarray:
    """
    Draw solved digits back onto the original image.
    
    Handles edge-to-edge grids by directly computing cell positions from corners.
    """
    result = original_image.copy()
    
    # Compute cell positions directly from corners (more robust for edge-to-edge grids)
    # corners: [top-left, top-right, bottom-right, bottom-left]
    tl = corners[0]
    tr = corners[1]
    br = corners[2]
    bl = corners[3]
    
    # For each cell, interpolate position from corners
    for row in range(9):
        for col in range(9):
            if original_puzzle[row, col] == 0 and solution[row, col] != 0:
                digit = solution[row, col]
                
                # Calculate normalized position (0 to 1) for cell center
                # Add 0.5 to get center of cell
                u = (col + 0.5) / 9.0
                v = (row + 0.5) / 9.0
                
                # Bilinear interpolation from corners
                # Top edge: interpolate between tl and tr
                top = tl + u * (tr - tl)
                # Bottom edge: interpolate between bl and br
                bottom = bl + u * (br - bl)
                # Final position: interpolate between top and bottom
                pos = top + v * (bottom - top)
                
                x, y = int(pos[0]), int(pos[1])
                
                # Calculate font scale based on cell size
                font_scale = estimate_font_scale(corners)
                
                # Get text size to center it
                text = str(digit)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale, thickness)[0]
                x -= text_size[0] // 2
                y += text_size[1] // 2
                
                # Draw text
                cv2.putText(result, text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return result


def draw_solution_on_warped(warped_image: np.ndarray, 
                            original_puzzle: np.ndarray, solution: np.ndarray,
                            color: Tuple[int, int, int] = (0, 180, 0),
                            thickness: int = 2) -> np.ndarray:
    """
    Draw solved digits on the warped (straightened) grid image.
    Useful when perspective transform back to original doesn't work well.
    """
    result = warped_image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    h, w = result.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    
    font_scale = min(cell_h, cell_w) / 40
    font_scale = max(0.5, min(font_scale, 2.0))
    
    for row in range(9):
        for col in range(9):
            if original_puzzle[row, col] == 0 and solution[row, col] != 0:
                digit = solution[row, col]
                
                # Cell center
                x = col * cell_w + cell_w // 2
                y = row * cell_h + cell_h // 2
                
                text = str(digit)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale, thickness)[0]
                x -= text_size[0] // 2
                y += text_size[1] // 2
                
                cv2.putText(result, text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return result
