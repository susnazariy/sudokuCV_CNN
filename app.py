"""Flask API server for Sudoku CV web interface."""

import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from sudoku_cv import SudokuPipeline, SudokuSolver
from sudoku_cv.solver import board_to_json

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Initialize pipeline (lazy loading)
pipeline = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global pipeline
    if pipeline is None:
        model_path = os.environ.get('MODEL_PATH', 'best_digit_model.pth')
        if os.path.exists(model_path):
            pipeline = SudokuPipeline(model_path)
        else:
            # Will need to train the model first
            pipeline = None
    return pipeline


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_base64_image(data_url):
    """Decode base64 image data URL to numpy array."""
    # Remove data URL prefix if present
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    
    # Decode base64
    img_bytes = base64.b64decode(data_url)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


def encode_image_base64(image):
    """Encode numpy array to base64 data URL."""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/api/solve/image', methods=['POST'])
def solve_from_image():
    """
    Solve Sudoku from uploaded image.
    
    Accepts:
        - File upload (multipart/form-data)
        - Base64 image (JSON with 'image' field)
    
    Returns:
        JSON with puzzle, solution, and result image
    """
    p = get_pipeline()
    if p is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    image = None
    
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Read image from file
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Handle base64 image
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            image = decode_base64_image(data['image'])
    
    if image is None:
        return jsonify({
            'success': False,
            'error': 'No valid image provided'
        }), 400
    
    # Process image
    result = p.process_image(image)
    
    response = {
        'success': result['success'],
        'error': result.get('error')
    }
    
    if result['success']:
        response['puzzle'] = board_to_json(result['puzzle'])
        response['solution'] = board_to_json(result['solution'])
        
        if result['result_image'] is not None:
            response['result_image'] = encode_image_base64(result['result_image'])
        
        # Also include warped result (better for edge-to-edge grids)
        if result.get('warped_result') is not None:
            response['warped_result'] = encode_image_base64(result['warped_result'])
    
    return jsonify(response)


@app.route('/api/solve/grid', methods=['POST'])
def solve_from_grid():
    """
    Solve Sudoku from manual grid input.
    
    Accepts JSON with 'grid' field (9x9 array, 0 for empty cells)
    
    Returns:
        JSON with solution
    """
    data = request.get_json()
    
    if not data or 'grid' not in data:
        return jsonify({
            'success': False,
            'error': 'No grid provided'
        }), 400
    
    try:
        grid = np.array(data['grid'], dtype=int)
        
        if grid.shape != (9, 9):
            return jsonify({
                'success': False,
                'error': 'Grid must be 9x9'
            }), 400
        
        solver = SudokuSolver(grid)
        success, solution = solver.solve()
        
        if success:
            return jsonify({
                'success': True,
                'solution': board_to_json(solution)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not solve the puzzle'
            })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_grid():
    """
    Validate a Sudoku grid.
    
    Accepts JSON with 'grid' field
    
    Returns:
        JSON with validation result
    """
    data = request.get_json()
    
    if not data or 'grid' not in data:
        return jsonify({
            'valid': False,
            'error': 'No grid provided'
        }), 400
    
    try:
        grid = np.array(data['grid'], dtype=int)
        
        if grid.shape != (9, 9):
            return jsonify({
                'valid': False,
                'error': 'Grid must be 9x9'
            }), 400
        
        solver = SudokuSolver(grid)
        
        return jsonify({
            'valid': True
        })
    
    except ValueError as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        })


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status and model availability."""
    p = get_pipeline()
    model_loaded = p is not None
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
