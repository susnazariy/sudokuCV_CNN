"""
Generate synthetic printed digit images for training.

This creates digit images that look like printed Sudoku numbers,
which differ from handwritten MNIST digits.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
import pickle


# Fonts prioritized for clean "1" without serifs/underscores
# Sans-serif fonts typically have simple "1" = just a vertical line
FONT_NAMES_PRIORITY = [
    # Sans-serif (clean "1" without underscore) - HIGHEST PRIORITY
    "DejaVuSans.ttf",
    "FreeSans.ttf",
    "LiberationSans-Regular.ttf",
    "NotoSans-Regular.ttf",
    "arial.ttf",
    "Arial.ttf",
    "helvetica.ttf",
    "Helvetica.ttf",
    "Verdana.ttf",
    "Tahoma.ttf",
    "Calibri.ttf",
    
    # Sans-serif Bold
    "DejaVuSans-Bold.ttf",
    "FreeSansBold.ttf",
    "LiberationSans-Bold.ttf",
    "arialbd.ttf",
    "Arial Bold.ttf",
    
    # Mono sans-serif
    "DejaVuSansMono.ttf",
    "FreeMono.ttf",
    "LiberationMono-Regular.ttf",
]

# Serif fonts (may have underscores on "1") - lower priority
FONT_NAMES_SERIF = [
    "DejaVuSerif.ttf",
    "FreeSerif.ttf",
    "LiberationSerif-Regular.ttf",
    "times.ttf",
    "Times.ttf",
    "TimesNewRoman.ttf",
    "Georgia.ttf",
]

# Common font directories
FONT_DIRS = [
    "/usr/share/fonts/truetype/",
    "/usr/share/fonts/",
    "/usr/local/share/fonts/",
    "C:/Windows/Fonts/",
    "/System/Library/Fonts/",
    "/Library/Fonts/",
    "~/.fonts/",
]


def find_available_fonts(min_fonts=3, serif_ratio=0.2):
    """
    Find available TrueType fonts on the system.
    Prioritizes sans-serif fonts (clean "1" without underscore).
    
    Args:
        min_fonts: Minimum fonts to find
        serif_ratio: Ratio of serif fonts to include (0.2 = 20%)
    """
    sans_serif = []
    serif = []
    
    font_dirs = [os.path.expanduser(d) for d in FONT_DIRS]
    
    def find_font(font_name):
        for font_dir in font_dirs:
            if not os.path.exists(font_dir):
                continue
            for root, dirs, files in os.walk(font_dir):
                if font_name in files:
                    return os.path.join(root, font_name)
        return None
    
    # Find sans-serif fonts (priority)
    for font_name in FONT_NAMES_PRIORITY:
        path = find_font(font_name)
        if path and path not in sans_serif:
            sans_serif.append(path)
    
    # Find serif fonts
    for font_name in FONT_NAMES_SERIF:
        path = find_font(font_name)
        if path and path not in serif:
            serif.append(path)
    
    # Combine with ratio (more sans-serif)
    available = sans_serif.copy()
    num_serif = max(1, int(len(sans_serif) * serif_ratio))
    available.extend(serif[:num_serif])
    
    print(f"Found {len(sans_serif)} sans-serif fonts (clean '1')")
    print(f"Found {len(serif)} serif fonts")
    print(f"Using {len(available)} total fonts ({len(sans_serif)} sans-serif + {min(num_serif, len(serif))} serif)")
    
    if len(available) < min_fonts:
        print(f"Warning: Only found {len(available)} fonts. Using default font as fallback.")
        available.append(None)
    
    return available, sans_serif


def generate_simple_one(size: int = 28, thickness: int = 3,
                        offset_x: int = 0, offset_y: int = 0,
                        rotation: float = 0.0, noise_level: float = 0.0) -> np.ndarray:
    """
    Generate a simple "1" as a vertical line (no serifs, no underscore).
    This matches how many printed Sudoku puzzles display "1".
    """
    render_size = size * 4
    img = Image.new('L', (render_size, render_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Calculate line dimensions
    line_height = int(render_size * 0.6)
    line_thickness = max(2, int(render_size * 0.08 * (1 + random.uniform(-0.3, 0.3))))
    
    # Center position with offset
    center_x = render_size // 2 + offset_x * 2
    center_y = render_size // 2 + offset_y * 2
    
    # Draw vertical line
    x1 = center_x - line_thickness // 2
    x2 = center_x + line_thickness // 2
    y1 = center_y - line_height // 2
    y2 = center_y + line_height // 2
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # Optional: add small top serif (like handwritten 1)
    if random.random() < 0.3:  # 30% chance
        serif_length = line_thickness * 2
        draw.line([(x1 - serif_length, y1 + line_thickness), (x1, y1)], fill=255, width=line_thickness)
    
    # Apply rotation
    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=0)
    
    # Resize to target size
    img = img.resize((size, size), resample=Image.LANCZOS)
    
    # Convert to numpy
    arr = np.array(img, dtype=np.float32)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randn(size, size) * noise_level * 255
        arr = np.clip(arr + noise, 0, 255)
    
    return arr.astype(np.uint8)


def generate_digit_image(digit: int, font_path: str = None, size: int = 28,
                         noise_level: float = 0.0, thickness_var: float = 0.0,
                         offset_x: int = 0, offset_y: int = 0,
                         rotation: float = 0.0, scale: float = 1.0,
                         invert: bool = True, use_simple_one: bool = False) -> np.ndarray:
    """
    Generate a single digit image.
    
    Args:
        digit: Digit to render (1-9)
        font_path: Path to TTF font file (None for default)
        size: Output image size (28 for MNIST compatibility)
        noise_level: Amount of random noise (0-1)
        thickness_var: Font size variation factor
        offset_x, offset_y: Position offset from center
        rotation: Rotation angle in degrees
        scale: Scale factor
        invert: If True, white digit on black (MNIST style)
        use_simple_one: If True and digit==1, draw simple vertical line
    
    Returns:
        28x28 numpy array (uint8)
    """
    # Special handling for digit "1" - use simple vertical line
    if digit == 1 and use_simple_one:
        return generate_simple_one(
            size=size,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation=rotation,
            noise_level=noise_level
        )
    # Create larger image for better quality, then resize
    render_size = size * 4
    img = Image.new('L', (render_size, render_size), color=0 if invert else 255)
    draw = ImageDraw.Draw(img)
    
    # Font size (base ~70% of image, with variation)
    base_font_size = int(render_size * 0.7 * scale)
    font_size = int(base_font_size * (1 + random.uniform(-thickness_var, thickness_var)))
    font_size = max(10, font_size)
    
    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
            # Scale up default font by drawing larger
            font_size = 20
    except Exception:
        font = ImageFont.load_default()
    
    # Get text bounding box
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center position with offset
    x = (render_size - text_width) // 2 + offset_x * 2
    y = (render_size - text_height) // 2 + offset_y * 2 - bbox[1]
    
    # Draw text
    fill_color = 255 if invert else 0
    draw.text((x, y), text, font=font, fill=fill_color)
    
    # Apply rotation
    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=0 if invert else 255)
    
    # Resize to target size
    img = img.resize((size, size), resample=Image.LANCZOS)
    
    # Convert to numpy
    arr = np.array(img, dtype=np.float32)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.randn(size, size) * noise_level * 255
        arr = np.clip(arr + noise, 0, 255)
    
    return arr.astype(np.uint8)


def generate_dataset(num_per_digit: int = 1000, output_dir: str = "printed_digits",
                     simple_one_ratio: float = 0.5):
    """
    Generate a complete dataset of printed digits.
    
    Args:
        num_per_digit: Number of images per digit (1-9)
        output_dir: Directory to save the dataset
        simple_one_ratio: Ratio of "1"s to draw as simple vertical line (0.5 = 50%)
    """
    print("Finding available fonts...")
    fonts, sans_serif_fonts = find_available_fonts()
    
    if not sans_serif_fonts:
        sans_serif_fonts = fonts  # Fallback
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Storage for all images and labels
    all_images = []
    all_labels = []
    
    # Generate for digits 1-9 (we don't need 0 for Sudoku)
    for digit in range(1, 10):
        print(f"Generating digit {digit}...")
        
        digit_images = []
        
        for i in range(num_per_digit):
            # Random augmentation parameters
            noise = random.uniform(0, 0.05)
            thickness = random.uniform(0, 0.2)
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            rotation = random.uniform(-10, 10)
            scale = random.uniform(0.8, 1.2)
            
            # Special handling for digit "1"
            if digit == 1:
                # 50% simple vertical line, 50% sans-serif font
                if random.random() < simple_one_ratio:
                    img = generate_digit_image(
                        digit=digit,
                        font_path=None,
                        noise_level=noise,
                        thickness_var=thickness,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        rotation=rotation,
                        scale=scale,
                        use_simple_one=True  # Draw as vertical line
                    )
                else:
                    # Use only sans-serif fonts for "1"
                    font = random.choice(sans_serif_fonts) if sans_serif_fonts else random.choice(fonts)
                    img = generate_digit_image(
                        digit=digit,
                        font_path=font,
                        noise_level=noise,
                        thickness_var=thickness,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        rotation=rotation,
                        scale=scale,
                        use_simple_one=False
                    )
            else:
                # Other digits: use all fonts
                font = random.choice(fonts)
                img = generate_digit_image(
                    digit=digit,
                    font_path=font,
                    noise_level=noise,
                    thickness_var=thickness,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    rotation=rotation,
                    scale=scale
                )
            
            digit_images.append(img)
            all_labels.append(digit)
        
        all_images.extend(digit_images)
        
        # Save sample images for verification
        sample_dir = output_path / "samples"
        sample_dir.mkdir(exist_ok=True)
        for j in range(min(10, len(digit_images))):
            img = Image.fromarray(digit_images[j])
            img.save(sample_dir / f"digit_{digit}_sample_{j}.png")
    
    # Convert to numpy arrays
    images = np.array(all_images, dtype=np.uint8)
    labels = np.array(all_labels, dtype=np.int64)
    
    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Split into train/test (80/20)
    split_idx = int(len(images) * 0.8)
    
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    test_images = images[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"\nDataset generated:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Testing: {len(test_images)} images")
    
    # Save as pickle (easy to load)
    dataset = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
    }
    
    with open(output_path / "printed_digits.pkl", 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSaved to {output_path / 'printed_digits.pkl'}")
    
    # Also save as individual numpy files
    np.save(output_path / "train_images.npy", train_images)
    np.save(output_path / "train_labels.npy", train_labels)
    np.save(output_path / "test_images.npy", test_images)
    np.save(output_path / "test_labels.npy", test_labels)
    
    print(f"Also saved as .npy files")
    
    return dataset


def visualize_samples(dataset_path: str = "printed_digits/printed_digits.pkl"):
    """Create a visualization grid of sample digits."""
    import matplotlib.pyplot as plt
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    images = dataset['train_images']
    labels = dataset['train_labels']
    
    # Show 9 rows (digits 1-9), 10 samples each
    fig, axes = plt.subplots(9, 10, figsize=(12, 12))
    
    for digit in range(1, 10):
        digit_indices = np.where(labels == digit)[0][:10]
        
        for j, idx in enumerate(digit_indices):
            ax = axes[digit - 1, j]
            ax.imshow(images[idx], cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(str(digit), fontsize=14, rotation=0, labelpad=20)
    
    plt.suptitle('Generated Printed Digits Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('printed_digits/samples_grid.png', dpi=150)
    plt.show()
    print("Saved samples_grid.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate printed digit dataset')
    parser.add_argument('--num', type=int, default=1000, help='Images per digit (default: 1000)')
    parser.add_argument('--output', type=str, default='printed_digits', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize after generation')
    
    args = parser.parse_args()
    
    dataset = generate_dataset(num_per_digit=args.num, output_dir=args.output)
    
    if args.visualize:
        try:
            visualize_samples(f"{args.output}/printed_digits.pkl")
        except ImportError:
            print("matplotlib not available for visualization")
