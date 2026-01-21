"""Sudoku puzzle solver using backtracking algorithm."""

import numpy as np
from typing import Tuple, Optional


class SudokuSolver:
    """Backtracking-based Sudoku solver."""
    
    def __init__(self, board: np.ndarray):
        if board.shape != (9, 9):
            raise ValueError(f"Board must be 9x9, got {board.shape}")
        
        self.original = board.astype(int).copy()
        self.board = board.astype(int).copy()
        
        if not self.is_valid_board():
            raise ValueError("Invalid board: contains duplicate numbers")
    
    def is_valid_board(self) -> bool:
        """Check if current board state is valid."""
        # Check rows
        for row in range(9):
            if not self._is_valid_group(self.board[row, :]):
                return False
        
        # Check columns
        for col in range(9):
            if not self._is_valid_group(self.board[:, col]):
                return False
        
        # Check 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = self.board[box_row:box_row+3, box_col:box_col+3].flatten()
                if not self._is_valid_group(box):
                    return False
        
        return True
    
    def _is_valid_group(self, group: np.ndarray) -> bool:
        """Check if a row/column/box has no duplicates."""
        nums = group[group > 0]
        return len(nums) == len(set(nums))
    
    def solve(self) -> Tuple[bool, np.ndarray]:
        """Solve the puzzle and return (success, solution)."""
        self.board = self.original.copy()
        
        if self._backtrack():
            return True, self.board.copy()
        else:
            return False, self.original.copy()
    
    def _backtrack(self) -> bool:
        """Recursive backtracking solver."""
        row, col = self._find_empty()
        
        if row is None:
            return True  # Solved
        
        for num in range(1, 10):
            if self._is_valid_placement(row, col, num):
                self.board[row, col] = num
                
                if self._backtrack():
                    return True
                
                self.board[row, col] = 0
        
        return False
    
    def _find_empty(self) -> Tuple[Optional[int], Optional[int]]:
        """Find the next empty cell."""
        for row in range(9):
            for col in range(9):
                if self.board[row, col] == 0:
                    return row, col
        return None, None
    
    def _is_valid_placement(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in self.board[row, :]:
            return False
        
        # Check column
        if num in self.board[:, col]:
            return False
        
        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        if num in self.board[box_row:box_row+3, box_col:box_col+3]:
            return False
        
        return True


def print_board(board: np.ndarray) -> str:
    """Format board as a string for display."""
    lines = []
    for i in range(9):
        if i % 3 == 0 and i != 0:
            lines.append("------+-------+------")
        
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            val = board[i, j]
            row_str += f"{val if val != 0 else '.'} "
        lines.append(row_str)
    
    return "\n".join(lines)


def board_to_json(board: np.ndarray) -> list:
    """Convert board to JSON-serializable format."""
    return board.tolist()


def json_to_board(data: list) -> np.ndarray:
    """Convert JSON data to numpy board."""
    return np.array(data, dtype=int)
