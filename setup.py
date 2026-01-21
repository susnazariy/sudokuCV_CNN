"""Setup script for Sudoku CV Solver."""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sudoku-cv',
    version='1.0.0',
    author='Your Name',
    description='Computer Vision Sudoku Solver using OpenCV and PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/sudoku-cv',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Games/Entertainment :: Puzzle Games',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'flask>=2.0.0',
    ],
    extras_require={
        'dev': [
            'matplotlib>=3.4.0',
            'jupyter>=1.0.0',
            'pytest>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sudoku-solve=solve:main',
            'sudoku-train=train:main',
        ],
    },
)
