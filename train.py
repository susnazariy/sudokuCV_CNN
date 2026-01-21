"""
Train the digit recognition model on MNIST dataset.

Usage:
    python train.py [--epochs N] [--lr RATE] [--output PATH]
"""

import argparse
from sudoku_cv.digit_recognition import DigitClassifier, load_mnist, train_model, device


def main():
    parser = argparse.ArgumentParser(description='Train digit recognition model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--output', type=str, default='best_digit_model.pth', help='Output model path')
    args = parser.parse_args()
    
    print(f"Training on device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("-" * 50)
    
    # Create and train model
    model = DigitClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)
    
    history = train_model(
        model, 
        train_loader, 
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.output
    )
    
    print("-" * 50)
    print(f"Training complete! Model saved to: {args.output}")
    
    # Print final stats
    print(f"\nFinal Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
