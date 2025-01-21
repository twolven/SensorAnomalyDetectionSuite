"""
Training script for power system pattern detection using YOLO.
Detects and localizes fault patterns in power system waveforms.
"""

import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaveformPatternDataset(Dataset):
    """Dataset class for waveform pattern detection."""
    
    def __init__(self, data, annotations, grid_size=32):
        self.data = torch.FloatTensor(data)
        self.grid_size = grid_size
        self.fault_types = ['sag', 'swell', 'interruption', 'harmonic']
        
        # Parse annotations and convert to YOLO format
        self.labels = []
        for ann_str in annotations:
            ann = json.loads(ann_str)
            grid_labels = np.zeros((grid_size, 5 + len(self.fault_types)))
            
            # Sort faults by start position for consistent processing
            faults = sorted(ann, key=lambda x: x['start'])
            
            for fault in faults:
                # Convert fault location to grid cell
                grid_x = int(fault['start'] * grid_size)
                grid_w = max(1, int(fault['width'] * grid_size))
                
                # One-hot encode fault type
                fault_type_idx = self.fault_types.index(fault['type'])
                
                # Fill grid cells covered by the fault
                for i in range(grid_w):
                    if grid_x + i < grid_size:
                        # Handle overlapping faults by preserving stronger fault signals
                        current_objectness = grid_labels[grid_x + i, 0]
                        if current_objectness < 1.0:  # If no strong fault already present
                            grid_labels[grid_x + i] = [
                                1.0,  # objectness
                                fault['start'] + i/grid_size,  # x center
                                0.5,  # y center (always centered in 1D)
                                fault['width'],  # width
                                fault['magnitude'],  # magnitude
                                *[1.0 if j == fault_type_idx else 0.0 
                                  for j in range(len(self.fault_types))]
                            ]
            
            self.labels.append(torch.FloatTensor(grid_labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class WaveformYOLO(nn.Module):
    """1D YOLO model for waveform pattern detection."""
    
    def __init__(self, n_fault_types, grid_size=32):
        super(WaveformYOLO, self).__init__()
        
        # Feature extractor with adaptive pooling
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Fourth conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Adaptive pooling to ensure correct output size
            nn.AdaptiveAvgPool1d(grid_size)
        )
        
        # Detection head
        self.detector = nn.Conv1d(256, 5 + n_fault_types, kernel_size=1)
        
        self.grid_size = grid_size
        self.n_fault_types = n_fault_types
        
    def forward(self, x):
        # Ensure input is in the right shape (batch, channels, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Pass through feature extractor
        features = self.features(x)
        
        # Apply detection head
        detections = self.detector(features)
        
        # Reshape to (batch, grid_size, 5 + n_fault_types)
        detections = detections.permute(0, 2, 1)  # (batch, grid, channels)
        
        return detections

class YOLOLoss(nn.Module):
    """Custom loss function for waveform YOLO."""
    
    def __init__(self, lambda_obj=15.0, lambda_noobj=0.1, lambda_coord=10.0, lambda_class=1.0):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')  # Changed to mean reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')  # Changed to mean reduction
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        
    def forward(self, predictions, targets):
        obj_mask = targets[..., 0] == 1
        noobj_mask = targets[..., 0] == 0
        
        # Objectness loss with focal loss characteristics
        obj_loss = self.bce(
            predictions[..., 0][obj_mask],
            targets[..., 0][obj_mask]
        )
        
        noobj_loss = self.bce(
            predictions[..., 0][noobj_mask],
            targets[..., 0][noobj_mask]
        )
        
        # Box coordinate and magnitude loss (only for objects)
        box_loss = self.mse(
            predictions[..., 1:5][obj_mask],
            targets[..., 1:5][obj_mask]
        )
        
        # Classification loss (only for objects)
        cls_loss = self.bce(
            predictions[..., 5:][obj_mask],
            targets[..., 5:][obj_mask]
        )
        
        total_loss = (
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_coord * box_loss +
            self.lambda_class * cls_loss
        )
        
        return total_loss

def load_data(data_path):
    """Load training and validation data."""
    logger.info(f"Loading data from {data_path}")
    with h5py.File(data_path, 'r') as f:
        x_train = f['train/data'][:]
        x_val = f['val/data'][:]
        ann_train = f['train/annotations'][:]
        ann_val = f['val/annotations'][:]
    
    # Normalize data
    max_val = np.max(np.abs(x_train))
    x_train = x_train.astype('float32') / max_val
    x_val = x_val.astype('float32') / max_val
    
    logger.info(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    return x_train, ann_train, x_val, ann_val

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    # Setup paths
    base_dir = Path("ml/training")
    data_path = base_dir / "data/yolo/pattern_detection.h5"
    model_path = base_dir / "models/yolo"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    x_train, ann_train, x_val, ann_val = load_data(data_path)
    
    # Create datasets
    train_dataset = WaveformPatternDataset(x_train, ann_train)
    val_dataset = WaveformPatternDataset(x_val, ann_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    
    # Create model
    logger.info("Creating model...")
    model = WaveformYOLO(n_fault_types=4).to(device)
    criterion = YOLOLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info("Training model...")
    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0
    training_history = []
    
    for epoch in range(100):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)
        
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'training_history': training_history
            }, model_path / 'waveform_detector.pth')
            patience_counter = 0
            logger.info(f'Saved new best model with validation loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    # Save training history
    with open(model_path / 'training_history.json', 'w') as f:
        json.dump(training_history, f)
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()