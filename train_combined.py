"""
Train digit recognition model with combined MNIST + printed digits.

Usage:
    python train_combined.py [--epochs N] [--printed-weight RATIO]
"""

import argparse
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

from sudoku_cv.digit_recognition import DigitClassifier, device


def load_printed_digits(path: str = "printed_digits/printed_digits.pkl"):
    """Load the generated printed digits dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Printed digits dataset not found at {path}.\n"
            f"Run: python generate_printed_digits.py --num 1000"
        )
    
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def create_combined_loaders(batch_size: int = 128, printed_path: str = "printed_digits/printed_digits.pkl",
                            mnist_ratio: float = 0.4):
    """
    Create data loaders combining MNIST and printed digits.
    
    Args:
        batch_size: Batch size for training
        printed_path: Path to printed digits dataset
        mnist_ratio: Ratio of MNIST in training (0.4 = 40% MNIST, 60% printed)
    """
    # MNIST transforms
    mnist_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    print("Loading MNIST dataset...")
    mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    # Filter MNIST to only include digits 1-9 (Sudoku doesn't use 0)
    mnist_train_indices = [i for i, (_, label) in enumerate(mnist_train_full) if label > 0]
    mnist_train = torch.utils.data.Subset(mnist_train_full, mnist_train_indices)
    
    # Load printed digits
    print("Loading printed digits dataset...")
    printed = load_printed_digits(printed_path)
    
    # Normalize printed digits the same way
    train_images = printed['train_images'].astype(np.float32) / 255.0
    train_images = (train_images - 0.1307) / 0.3081
    train_labels = printed['train_labels'].astype(np.int64)
    
    test_images = printed['test_images'].astype(np.float32) / 255.0
    test_images = (test_images - 0.1307) / 0.3081
    test_labels = printed['test_labels'].astype(np.int64)
    
    # Add channel dimension
    train_images = train_images[:, np.newaxis, :, :]
    test_images = test_images[:, np.newaxis, :, :]
    
    # Create tensor datasets
    printed_train = TensorDataset(
        torch.from_numpy(train_images).float(),
        torch.from_numpy(train_labels).long()
    )
    printed_test = TensorDataset(
        torch.from_numpy(test_images).float(),
        torch.from_numpy(test_labels).long()
    )
    
    print(f"MNIST train (digits 1-9): {len(mnist_train)} images")
    print(f"Printed train: {len(printed_train)} images")
    
    # Calculate how many times to repeat printed to achieve desired ratio
    # We want: mnist_ratio = len(mnist) / (len(mnist) + len(printed_repeated))
    # So: printed_repeated = len(mnist) * (1 - mnist_ratio) / mnist_ratio
    num_mnist = len(mnist_train)
    target_printed = int(num_mnist * (1 - mnist_ratio) / mnist_ratio)
    
    # Calculate repetitions needed for printed dataset
    repeat_times = max(1, target_printed // len(printed_train) + 1)
    repeated_printed = ConcatDataset([printed_train] * repeat_times)
    
    # Subsample repeated printed to exact size needed
    if len(repeated_printed) > target_printed:
        printed_subset_indices = random.sample(range(len(repeated_printed)), target_printed)
        repeated_printed = torch.utils.data.Subset(repeated_printed, printed_subset_indices)
    
    print(f"After balancing (target {mnist_ratio:.0%} MNIST / {1-mnist_ratio:.0%} printed):")
    print(f"  MNIST (full): {num_mnist} images")
    print(f"  Printed (repeated): {len(repeated_printed)} images")
    
    # Combine datasets
    combined_train = ConcatDataset([mnist_train, repeated_printed])
    combined_test = ConcatDataset([mnist_test, printed_test])
    
    actual_mnist_ratio = num_mnist / len(combined_train)
    print(f"  Combined train: {len(combined_train)} images ({actual_mnist_ratio:.1%} MNIST, {1-actual_mnist_ratio:.1%} printed)")
    print(f"  Combined test: {len(combined_test)} images")
    
    # Custom collate function to ensure consistent types
    def custom_collate(batch):
        images = []
        labels = []
        for img, label in batch:
            images.append(img)
            if isinstance(label, int):
                labels.append(torch.tensor(label, dtype=torch.long))
            else:
                labels.append(label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long))
        return torch.stack(images), torch.stack(labels)
    
    # Create loaders with custom collate
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Printed-only test loader
    printed_test_loader = DataLoader(printed_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, printed_test_loader


def evaluate(model, loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0


def train_combined(epochs: int = 10, lr: float = 0.001, batch_size: int = 128,
                   printed_path: str = "printed_digits/printed_digits.pkl",
                   output_path: str = "best_digit_model.pth"):
    """Train model on combined dataset."""
    
    print(f"Training on device: {device}")
    print("-" * 50)
    
    # Load data
    train_loader, test_loader, printed_test_loader = create_combined_loaders(
        batch_size=batch_size,
        printed_path=printed_path
    )
    print("-" * 50)
    
    # Create model
    model = DigitClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    best_acc = 0
    best_printed_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation
        val_acc = evaluate(model, test_loader, device)
        printed_acc = evaluate(model, printed_test_loader, device)
        
        scheduler.step(1 - val_acc)
        
        # Save best model (40% weight on general accuracy, 60% on printed)
        combined_score = 0.4 * val_acc + 0.6 * printed_acc
        if combined_score > best_acc:
            best_acc = combined_score
            best_printed_acc = printed_acc
            torch.save(model.state_dict(), output_path)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Printed Acc: {printed_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(output_path, map_location=device))
    
    print("-" * 50)
    print(f"Training complete!")
    print(f"Best combined score: {best_acc:.4f}")
    print(f"Best printed digits accuracy: {best_printed_acc:.4f}")
    print(f"Model saved to: {output_path}")
    
    # Final evaluation breakdown
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    final_val_acc = evaluate(model, test_loader, device)
    final_printed_acc = evaluate(model, printed_test_loader, device)
    
    print(f"Combined test accuracy: {final_val_acc:.4f}")
    print(f"Printed digits accuracy: {final_printed_acc:.4f}")
    
    # Per-digit accuracy on printed digits
    print("\nPer-digit accuracy (printed):")
    model.eval()
    digit_correct = {i: 0 for i in range(1, 10)}
    digit_total = {i: 0 for i in range(1, 10)}
    
    with torch.no_grad():
        for data, target in printed_test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for p, t in zip(pred, target):
                digit = t.item()
                if digit > 0:
                    digit_total[digit] += 1
                    if p.item() == digit:
                        digit_correct[digit] += 1
    
    for digit in range(1, 10):
        if digit_total[digit] > 0:
            acc = digit_correct[digit] / digit_total[digit]
            print(f"  Digit {digit}: {acc:.4f} ({digit_correct[digit]}/{digit_total[digit]})")


def main():
    parser = argparse.ArgumentParser(description='Train with combined MNIST + printed digits')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--printed-path', type=str, default='printed_digits/printed_digits.pkl',
                        help='Path to printed digits dataset')
    parser.add_argument('--output', type=str, default='best_digit_model.pth', help='Output model path')
    
    args = parser.parse_args()
    
    train_combined(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        printed_path=args.printed_path,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
