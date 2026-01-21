"""
Command-line interface for Sudoku CV Solver.

Usage:
    python solve.py image.png [-o output.png]
    python solve.py --grid "530070000,600195000,..."
"""

import argparse
import sys
import numpy as np

from sudoku_cv import SudokuPipeline, SudokuSolver, print_board


def parse_grid_string(grid_str: str) -> np.ndarray:
    """Parse grid from comma-separated rows or single string."""
    grid_str = grid_str.replace(' ', '').replace('\n', '')
    
    if ',' in grid_str:
        rows = grid_str.split(',')
    else:
        # Single string of 81 digits
        if len(grid_str) != 81:
            raise ValueError(f"Grid string must have 81 digits, got {len(grid_str)}")
        rows = [grid_str[i:i+9] for i in range(0, 81, 9)]
    
    if len(rows) != 9:
        raise ValueError(f"Grid must have 9 rows, got {len(rows)}")
    
    grid = []
    for row in rows:
        if len(row) != 9:
            raise ValueError(f"Each row must have 9 digits, got {len(row)}")
        grid.append([int(c) if c.isdigit() else 0 for c in row])
    
    return np.array(grid)


def solve_image(image_path: str, model_path: str, output_path: str = None):
    """Solve Sudoku from image file."""
    print(f"Loading image: {image_path}")
    
    pipeline = SudokuPipeline(model_path)
    result = pipeline.process_file(image_path)
    
    if not result['success']:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1
    
    print("\n📷 Detected Puzzle:")
    print(print_board(result['puzzle']))
    
    print("\n✅ Solution:")
    print(print_board(result['solution']))
    
    if output_path:
        import cv2
        cv2.imwrite(output_path, result['result_image'])
        print(f"\n💾 Result saved to: {output_path}")
    
    return 0


def solve_grid(grid_str: str):
    """Solve Sudoku from grid string."""
    try:
        grid = parse_grid_string(grid_str)
    except ValueError as e:
        print(f"Error parsing grid: {e}", file=sys.stderr)
        return 1
    
    print("📝 Input Puzzle:")
    print(print_board(grid))
    
    try:
        solver = SudokuSolver(grid)
        success, solution = solver.solve()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if success:
        print("\n✅ Solution:")
        print(print_board(solution))
        return 0
    else:
        print("Error: Could not solve the puzzle", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Solve Sudoku puzzles from images or text input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sudoku.png
  %(prog)s sudoku.png -o solved.png
  %(prog)s --grid "530070000,600195000,098000060,800060003,400803001,700020006,060000280,000419005,000080079"
  %(prog)s --grid "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        """
    )
    
    parser.add_argument('image', nargs='?', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('-m', '--model', default='best_digit_model.pth', help='Model path')
    parser.add_argument('--grid', help='Solve from grid string (comma-separated rows or 81 digits)')
    
    args = parser.parse_args()
    
    if args.grid:
        sys.exit(solve_grid(args.grid))
    elif args.image:
        sys.exit(solve_image(args.image, args.model, args.output))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
