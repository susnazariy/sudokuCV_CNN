"""Digit recognition using PyTorch CNN trained on MNIST."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict


# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DigitClassifier(nn.Module):
    """CNN for digit classification (0-9)."""
    
    def __init__(self):
        super(DigitClassifier, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully connected layers (28 -> 14 -> 7, so 7*7*64 = 3136)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def load_mnist(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset with augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                epochs: int = 5, lr: float = 0.01, save_path: str = 'best_digit_model.pth') -> Dict:
    """Train the digit classifier."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    
    for epoch in range(epochs):
        # Training phase
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(test_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    
    return history


class DigitRecognizer:
    """High-level interface for digit recognition."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = DigitClassifier()
        self.device = device
        
        if model_path:
            self.load(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # MNIST normalization
        self.mean = 0.1307
        self.std = 0.3081
    
    def load(self, path: str) -> None:
        """Load trained weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def preprocess(self, cell: np.ndarray) -> torch.Tensor:
        """Preprocess cell image for the model with better centering."""
        import cv2
        
        # Find bounding box of the digit
        coords = cv2.findNonZero(cell)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Extract digit with small margin
            margin = 2
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(cell.shape[1], x + w + margin)
            y2 = min(cell.shape[0], y + h + margin)
            
            digit = cell[y1:y2, x1:x2]
            
            # Make it square by padding shorter side
            h_digit, w_digit = digit.shape
            if h_digit > w_digit:
                pad = (h_digit - w_digit) // 2
                digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
            elif w_digit > h_digit:
                pad = (w_digit - h_digit) // 2
                digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
            # Resize to 20x20 (MNIST style - digits are ~20x20 centered in 28x28)
            digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
            
            # Center in 28x28 image
            centered = np.zeros((28, 28), dtype=np.uint8)
            centered[4:24, 4:24] = digit
            cell = centered
        
        img = cell.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, cell: np.ndarray) -> Tuple[int, float]:
        """Predict digit from a single cell."""
        tensor = self.preprocess(cell)
        
        with torch.no_grad():
            probs = self.model.predict_proba(tensor)[0]
            digit = probs.argmax().item()
            confidence = probs[digit].item()
        
        # Disambiguation for 1 vs 7 confusion
        if digit in [1, 7] and confidence < 0.9:
            digit = self._disambiguate_1_and_7(cell, probs)
            confidence = probs[digit].item()
        
        return digit, confidence
    
    def _disambiguate_1_and_7(self, cell: np.ndarray, probs: torch.Tensor) -> int:
        """Use shape analysis to distinguish 1 from 7."""
        import cv2
        
        # Find bounding box
        coords = cv2.findNonZero(cell)
        if coords is None:
            return probs.argmax().item()
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extract the digit region
        digit_region = cell[y:y+h, x:x+w]
        
        if digit_region.size == 0:
            return probs.argmax().item()
        
        # Feature 1: Aspect ratio (7 is usually wider than 1)
        aspect_ratio = w / max(h, 1)
        
        # Feature 2: Check for horizontal stroke at top (7 has it, 1 doesn't)
        top_third = digit_region[:max(h//3, 1), :]
        top_density = np.sum(top_third > 128) / max(top_third.size, 1)
        
        # Feature 3: Check width of top vs bottom (7 wider at top, 1 uniform)
        top_row_width = np.sum(digit_region[:max(h//4, 1), :] > 128)
        bottom_row_width = np.sum(digit_region[-(max(h//4, 1)):, :] > 128)
        width_ratio = top_row_width / max(bottom_row_width, 1)
        
        # Decision logic
        # "7" characteristics: wider aspect ratio, dense top, wider at top
        # "1" characteristics: narrow aspect ratio, sparse top, uniform width
        
        score_for_7 = 0
        score_for_1 = 0
        
        if aspect_ratio > 0.4:
            score_for_7 += 1
        else:
            score_for_1 += 1
        
        if top_density > 0.3:
            score_for_7 += 1
        else:
            score_for_1 += 1
        
        if width_ratio > 1.5:
            score_for_7 += 1
        else:
            score_for_1 += 1
        
        # Combine with neural network confidence
        prob_1 = probs[1].item()
        prob_7 = probs[7].item()
        
        # Weighted decision
        final_score_1 = prob_1 + (score_for_1 * 0.15)
        final_score_7 = prob_7 + (score_for_7 * 0.15)
        
        return 1 if final_score_1 > final_score_7 else 7
    
    def predict_batch(self, cells: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Predict digits from multiple cells."""
        return [self.predict(cell) for cell in cells]
    
    def predict_sudoku(self, cells: List[np.ndarray], cell_status: List[bool],
                       empty_threshold: float = 0) -> np.ndarray:
        """Predict the entire Sudoku grid."""
        predictions = []
        
        for cell, has_digit in zip(cells, cell_status):
            if not has_digit:
                predictions.append(0)
            else:
                digit, confidence = self.predict(cell)
                
                if confidence < empty_threshold or digit == 0:
                    predictions.append(0)
                else:
                    predictions.append(digit)
        
        return np.array(predictions).reshape(9, 9)
