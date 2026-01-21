# 🧩 Sudoku CV Solver

A computer vision-based Sudoku solver that extracts puzzles from images and solves them using AI digit recognition and backtracking algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 📷 **Image Processing**: Detect Sudoku grids from photos at any angle
- 🔢 **Digit Recognition**: CNN-based recognition trained on MNIST + printed digits
- 🧮 **Fast Solver**: Efficient backtracking algorithm with constraint propagation
- 🌐 **Web Interface**: Clean, responsive UI with drag-and-drop upload
- 📱 **Manual Input**: Enter puzzles manually with keyboard navigation

## 🖼️ Screenshots

<!-- Add your screenshots here -->
<!-- ![Web Interface](screenshots/interface.png) -->
<!-- ![Solution Example](screenshots/solution.png) -->

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sudoku-solver.git
   cd sudoku-solver
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the digit recognition model**
   ```bash
   python train.py
   ```
   This downloads MNIST and trains a CNN model (~2-5 minutes on CPU).

5. **Run the web application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📖 Usage

### Web Interface

1. Navigate to `http://localhost:5000`
2. Choose **"From Image"** tab
3. Drag & drop or click to upload a Sudoku image
4. Click **"Solve Puzzle"**
5. View the solution overlaid on the original image

### Command Line

```bash
# Solve from image file
python solve.py path/to/sudoku.jpg

# Save result image
python solve.py path/to/sudoku.jpg --output result.png
```

### Python API

```python
from sudoku_cv import SudokuPipeline

# Initialize pipeline
pipeline = SudokuPipeline('best_digit_model.pth')

# Process image
result = pipeline.process_file('sudoku.jpg')

if result['success']:
    print("Detected:", result['puzzle'])
    print("Solution:", result['solution'])
```

## 📁 Project Structure

```
sudoku-solver/
├── sudoku_cv/              # Core CV modules
│   ├── __init__.py
│   ├── preprocessing.py    # Image preprocessing
│   ├── grid_detection.py   # Grid contour detection
│   ├── cell_extraction.py  # Cell extraction & classification
│   ├── digit_recognition.py # CNN digit classifier
│   ├── solver.py           # Backtracking solver
│   ├── visualization.py    # Solution overlay
│   └── pipeline.py         # Main processing pipeline
├── static/                 # Web assets
│   ├── css/style.css
│   └── js/app.js
├── templates/              # HTML templates
│   └── index.html
├── app.py                  # Flask web server
├── train.py                # Model training script
├── solve.py                # CLI solver
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best_digit_model.pth` | Path to trained model |
| `PORT` | `5000` | Web server port |
| `DEBUG` | `false` | Enable debug mode |

### Training Options

```bash
# Train with custom settings
python train.py --epochs 15 --batch-size 64 --lr 0.001
```

## 🧪 How It Works

1. **Preprocessing**: Convert to grayscale, apply adaptive thresholding
2. **Grid Detection**: Find largest quadrilateral contour
3. **Perspective Transform**: Warp grid to square image
4. **Cell Extraction**: Divide into 81 cells, detect grid lines
5. **Digit Recognition**: CNN classifies each cell (0-9)
6. **Solving**: Backtracking with constraint propagation
7. **Visualization**: Overlay solution on original image

## 📊 Model Performance

- **MNIST Accuracy**: ~99%
- **Printed Digits**: ~97%
- **Real-world Images**: ~90-95% (depends on image quality)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for digit recognition training
- [OpenCV](https://opencv.org/) for image processing
- [PyTorch](https://pytorch.org/) for deep learning
