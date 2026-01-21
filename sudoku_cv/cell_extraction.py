"""Cell extraction and classification utilities."""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def detect_grid_lines(warped: np.ndarray) -> Tuple[List[int], List[int]]:
    """Detect horizontal and vertical grid lines using morphology."""
    h, w = warped.shape[:2]
    
    # Create kernels to detect lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 9, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 9))
    
    # Detect horizontal lines
    horizontal = cv2.morphologyEx(warped, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical = cv2.morphologyEx(warped, cv2.MORPH_OPEN, vertical_kernel)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=1)
    
    # Find line positions by summing along axes
    h_sum = np.sum(horizontal, axis=1)
    v_sum = np.sum(vertical, axis=0)
    
    h_lines = find_line_positions(h_sum, 10)
    v_lines = find_line_positions(v_sum, 10)
    
    return h_lines, v_lines


def find_line_positions(profile: np.ndarray, num_lines: int, 
                        min_distance: int = 20) -> List[int]:
    """Find line positions from intensity profile."""
    threshold = np.max(profile) * 0.3
    peaks = []
    
    in_peak = False
    peak_start = 0
    
    for i, val in enumerate(profile):
        if val > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif val <= threshold and in_peak:
            in_peak = False
            peak_center = (peak_start + i) // 2
            
            if not peaks or (peak_center - peaks[-1]) > min_distance:
                peaks.append(peak_center)
    
    # Fallback to uniform spacing if detection fails
    if len(peaks) != num_lines:
        size = len(profile)
        peaks = [int(i * size / (num_lines - 1)) for i in range(num_lines)]
    
    return sorted(peaks)


def extract_cells_adaptive(warped: np.ndarray, 
                          cell_size: int = 28) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Extract 81 cells from the warped grid image."""
    h_lines, v_lines = detect_grid_lines(warped)
    cells = []
    
    for row in range(9):
        for col in range(9):
            y1 = h_lines[row]
            y2 = h_lines[row + 1]
            x1 = v_lines[col]
            x2 = v_lines[col + 1]
            
            cell = warped[y1:y2, x1:x2]
            
            if cell.size > 0:
                cell = cv2.resize(cell, (cell_size, cell_size))
            else:
                cell = np.zeros((cell_size, cell_size), dtype=np.uint8)
            
            cells.append(cell)
    
    return cells, h_lines, v_lines


def find_largest_contour(cell: np.ndarray) -> np.ndarray:
    """Keep only the largest contour in a cell (the digit)."""
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(cell)
    
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < 15:
        return np.zeros_like(cell)
    
    mask = np.zeros_like(cell)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    return cv2.bitwise_and(cell, mask)


def add_black_frame(img: np.ndarray, thickness: int) -> np.ndarray:
    """Add a black frame to mask cell borders."""
    h, w = img.shape[:2]
    framed = img.copy()
    cv2.rectangle(framed, (0, 0), (w - 1, h - 1), color=0, thickness=thickness)
    return framed


def extract_cell_features(cell: np.ndarray) -> Dict[str, float]:
    """Extract features for blank cell classification."""
    h, w = cell.shape
    
    total_white = np.sum(cell > 128)
    
    margin = h // 4
    center = cell[margin:h-margin, margin:w-margin]
    center_white = np.sum(center > 128)
    
    center_ratio = center_white / max(total_white, 1)
    
    _, binary = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    
    if num_labels > 1:
        max_component_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    else:
        max_component_area = 0
    
    if num_labels > 1:
        max_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        comp_w = stats[max_idx, cv2.CC_STAT_WIDTH]
        comp_h = stats[max_idx, cv2.CC_STAT_HEIGHT]
        density = max_component_area / max(comp_w * comp_h, 1)
    else:
        density = 0
    
    return {
        'total_white': total_white,
        'center_white': center_white,
        'center_ratio': center_ratio,
        'component_area': max_component_area,
        'density': density
    }


def find_gap_threshold(values: np.ndarray) -> float:
    """Find optimal threshold using largest gap in sorted values."""
    sorted_vals = np.sort(values)
    gaps = np.diff(sorted_vals)
    
    if len(gaps) == 0:
        return np.median(values)
    
    max_gap_idx = np.argmax(gaps)
    threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2
    
    return threshold


def find_blank_cells_adaptive(cells: List[np.ndarray]) -> Tuple[List[bool], Dict[str, float]]:
    """Classify cells as blank or containing digits."""
    all_features = [extract_cell_features(cell) for cell in cells]
    
    total_whites = np.array([f['total_white'] for f in all_features])
    center_whites = np.array([f['center_white'] for f in all_features])
    component_areas = np.array([f['component_area'] for f in all_features])
    
    thresholds = {
        'total_white': find_gap_threshold(total_whites),
        'center_white': find_gap_threshold(center_whites),
        'component_area': find_gap_threshold(component_areas)
    }
    
    cell_status = []
    for f in all_features:
        has_digit = (
            f['total_white'] > thresholds['total_white'] and
            f['center_white'] > thresholds['center_white'] and
            f['component_area'] > thresholds['component_area']
        )
        cell_status.append(has_digit)
    
    return cell_status, thresholds
