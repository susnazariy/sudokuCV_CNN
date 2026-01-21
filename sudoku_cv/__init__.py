"""
Sudoku CV - Computer Vision Sudoku Solver

A complete pipeline to solve Sudoku puzzles from images using:
- OpenCV for image processing and grid detection
- PyTorch CNN for digit recognition
- Backtracking algorithm for puzzle solving
"""

from .pipeline import SudokuPipeline, solve_from_image
from .solver import SudokuSolver, print_board
from .digit_recognition import DigitRecognizer, DigitClassifier, train_model, load_mnist

__version__ = '1.0.0'
__all__ = [
    'SudokuPipeline',
    'solve_from_image',
    'SudokuSolver',
    'print_board',
    'DigitRecognizer',
    'DigitClassifier',
    'train_model',
    'load_mnist',
]
