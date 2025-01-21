"""
Validation script for fault classification data with minimal model
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

def plot_samples(x_data, y_data, fault_types, save_path):
    """Plot sample waveforms from each class"""
    plt.figure(figsize=(15, 10))
    for i in range(len(fault_types)):
        class_samples = x_data[y_data == i]
        if len(class_samples) > 0:
            plt.subplot(len(fault_types), 1, i+1)
            for j in range(min(3, len(class_samples))):
                plt.plot(class_samples[j], alpha=0.7, label=f'Sample {j+1}')
            plt.title(f"Class: {fault_types[i]}")
            plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        # Simple standardization
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SimpleClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Force output size
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    print("\n=== Validating Fault Classification Data ===")
    
    # Setup paths
    base_dir = Path("ml/training")
    data_path = base_dir / "data/pytorch/fault_classification.h5"
    
    # Load and examine data
    print("\nLoading data...")
    with h5py.File(data_path, 'r') as f:
        x_train = f['train/data'][:]
        y_train = f['train/labels'][:]
        x_val = f['val/data'][:]
        y_val = f['val/labels'][:]
        fault_types = [x.decode('utf-8') for x in f['fault_types'][:]]
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print("\nClass distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"{fault_types[label]}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Check for data issues
    print("\nData Validation:")
    print(f"NaN values in training: {np.isnan(x_train).any()}")
    print(f"Inf values in training: {np.isinf(x_train).any()}")
    print(f"Zero variance sequences in training: {np.any(np.std(x_train, axis=1) == 0)}")
    print(f"\nSignal statistics:")
    print(f"Mean amplitude: {np.mean(np.abs(x_train)):.3f}")
    print(f"Std amplitude: {np.std(x_train):.3f}")
    print(f"Max amplitude: {np.max(np.abs(x_train)):.3f}")
    print(f"Min amplitude: {np.min(x_train):.3f}")
    
    # Plot sample waveforms
    plot_samples(x_train, y_train, fault_types, base_dir / 'sample_waveforms.png')
    
    # Create datasets
    train_dataset = SimpleDataset(x_train, y_train)
    val_dataset = SimpleDataset(x_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier(len(fault_types)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nStarting validation training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Acc = {100.*correct/total:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = np.zeros(len(fault_types))
        class_total = np.zeros(len(fault_types))
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Per-class accuracy
                for i in range(len(fault_types)):
                    mask = targets == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (predicted[mask] == i).sum().item()
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Training accuracy: {100.*correct/total:.2f}%")
        print(f"Validation accuracy: {100.*val_correct/val_total:.2f}%")
        print("\nPer-class validation accuracy:")
        for i in range(len(fault_types)):
            if class_total[i] > 0:
                print(f"{fault_types[i]}: {100.*class_correct[i]/class_total[i]:.2f}%")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()