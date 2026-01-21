"""Main pipeline for Sudoku solving from images."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .preprocessing import preprocess_image
from .grid_detection import find_grid_contour, order_corners, warp_perspective
from .cell_extraction import (
    extract_cells_adaptive, add_black_frame, 
    find_largest_contour, find_blank_cells_adaptive
)
from .digit_recognition import DigitRecognizer
from .solver import SudokuSolver, print_board
from .visualization import draw_solution_on_image, draw_solution_on_warped


class SudokuPipeline:
    """Complete Sudoku solving pipeline."""
    
    def __init__(self, model_path: str = 'best_digit_model.pth', grid_size: int = 450):
        self.model_path = model_path
        self.grid_size = grid_size
        self.recognizer = None  # Lazy loading
    
    def _ensure_recognizer(self):
        """Lazy load the digit recognizer."""
        if self.recognizer is None:
            self.recognizer = DigitRecognizer(self.model_path)
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a Sudoku image and return results.
        
        Returns:
            Dict with keys:
                - success: bool
                - puzzle: 9x9 array of detected digits
                - solution: 9x9 array of solved puzzle
                - result_image: image with solution overlay on original
                - warped_result: solution on warped grid (better for edge-to-edge images)
                - error: error message if failed
        """
        result = {
            'success': False,
            'puzzle': None,
            'solution': None,
            'result_image': None,
            'warped_result': None,
            'error': None
        }
        
        try:
            # Preprocess
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            thresh = preprocess_image(blurred)
            
            # Find grid (tries with white padding if initial detection fails)
            grid_contour = find_grid_contour(thresh, thresh.shape)
            
            if grid_contour is None:
                result['error'] = "Could not detect Sudoku grid in image"
                return result
            
            corners = order_corners(grid_contour)
            
            # Warp perspective
            warped, matrix = warp_perspective(image, corners, self.grid_size)
            thresh_warped = preprocess_image(warped)
            
            # Extract cells
            cells, h_lines, v_lines = extract_cells_adaptive(thresh_warped)
            
            # Clean cells
            thickness = int(cells[0].shape[0] * 0.1)
            cells = [add_black_frame(cell, thickness) for cell in cells]
            cells = [find_largest_contour(cell) for cell in cells]
            cells = [cv2.GaussianBlur(cell, (3, 3), 0) for cell in cells]
            
            # Classify blank cells
            cell_status, _ = find_blank_cells_adaptive(cells)
            
            # Recognize digits
            self._ensure_recognizer()
            puzzle = self.recognizer.predict_sudoku(cells, cell_status)
            result['puzzle'] = puzzle
            
            # Solve
            try:
                solver = SudokuSolver(puzzle)
                success, solution = solver.solve()
            except ValueError as e:
                result['error'] = f"Invalid puzzle: {str(e)}"
                return result
            
            if not success:
                result['error'] = "Could not solve the puzzle"
                return result
            
            result['solution'] = solution
            
            # Draw solution on original image
            result_image = draw_solution_on_image(
                image, corners, puzzle, solution, self.grid_size
            )
            result['result_image'] = result_image
            
            # Also draw on warped image (more reliable for edge-to-edge grids)
            warped_result = draw_solution_on_warped(warped, puzzle, solution)
            result['warped_result'] = warped_result
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = f"Processing error: {str(e)}"
        
        return result
    
    def process_file(self, path: str) -> Dict[str, Any]:
        """Process a Sudoku image from file path."""
        image = cv2.imread(path)
        if image is None:
            return {
                'success': False,
                'error': f"Could not load image: {path}"
            }
        return self.process_image(image)


def solve_from_image(image_path: str, model_path: str = 'best_digit_model.pth',
                     output_path: Optional[str] = None) -> Tuple[bool, np.ndarray]:
    """
    Convenience function to solve a Sudoku from an image file.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model weights
        output_path: Optional path to save result image
    
    Returns:
        (success, solution_array)
    """
    pipeline = SudokuPipeline(model_path)
    result = pipeline.process_file(image_path)
    
    if result['success']:
        print("Detected puzzle:")
        print(print_board(result['puzzle']))
        print("\nSolution:")
        print(print_board(result['solution']))
        
        if output_path and result['result_image'] is not None:
            cv2.imwrite(output_path, result['result_image'])
            print(f"\nResult saved to: {output_path}")
        
        return True, result['solution']
    else:
        print(f"Error: {result['error']}")
        return False, None
