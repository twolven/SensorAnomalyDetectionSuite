"""
Improved fault classifier using specialized feature engineering and ensemble of binary classifiers.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal, stats

class FeatureExtractor:
    """Extract specialized features for each type of power system fault."""
    
    @staticmethod
    def get_features(waveform):
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(waveform),
            np.std(waveform),
            np.median(waveform),
            stats.skew(waveform),
            stats.kurtosis(waveform)
        ])
        
        # RMS and crest features
        rms = np.sqrt(np.mean(np.square(waveform)))
        peak = np.max(np.abs(waveform))
        features.extend([
            rms,
            peak,
            peak/rms if rms > 0 else 0,  # Crest factor
        ])
        
        # Sag/Swell detection features
        window_size = min(60, len(waveform))
        rolling_rms = np.array([
            np.sqrt(np.mean(np.square(waveform[i:i+window_size])))
            for i in range(0, len(waveform)-window_size, window_size)
        ])
        features.extend([
            np.min(rolling_rms)/np.mean(rolling_rms) if np.mean(rolling_rms) > 0 else 0,  # Sag depth
            np.max(rolling_rms)/np.mean(rolling_rms) if np.mean(rolling_rms) > 0 else 0,  # Swell height
            np.std(rolling_rms)/np.mean(rolling_rms) if np.mean(rolling_rms) > 0 else 0   # Variation
        ])
        
        # Frequency domain features
        freqs, psd = signal.welch(waveform, fs=60, nperseg=min(256, len(waveform)))
        fundamental_idx = np.argmax(psd)
        harmonic_power = np.sum(psd[fundamental_idx+1:])
        total_power = np.sum(psd)
        
        features.extend([
            harmonic_power/psd[fundamental_idx] if psd[fundamental_idx] > 0 else 0,  # THD estimate
            np.sum(psd[1:6])/total_power if total_power > 0 else 0,  # Low order harmonics
            np.sum(psd[6:])/total_power if total_power > 0 else 0,   # High order harmonics
        ])
        
        # Interruption features
        zero_regions = np.where(np.abs(waveform) < 0.1 * rms)[0]
        if len(zero_regions) > 0:
            max_interruption = np.max(np.diff(np.where(np.abs(waveform) >= 0.1 * rms)[0]))
            features.append(max_interruption/len(waveform))
        else:
            features.append(0)
            
        # Rate of change features
        diffs = np.diff(waveform)
        features.extend([
            np.mean(np.abs(diffs)),
            np.std(diffs),
            np.max(np.abs(diffs))
        ])
        
        # Shape features
        zero_crossings = np.sum(np.diff(np.signbit(waveform).astype(int)))
        features.extend([
            zero_crossings,
            len(signal.find_peaks(waveform)[0])/len(waveform)
        ])
        
        # Symmetry features
        half_len = len(waveform)//2
        features.extend([
            np.corrcoef(waveform[:half_len], waveform[half_len:2*half_len])[0,1],
            np.mean(np.abs(waveform[:half_len] + waveform[half_len:2*half_len]))
        ])
        
        return np.array(features, dtype=np.float32)

class BinaryClassifier(nn.Module):
    """Simple classifier for binary fault detection."""
    
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class PowerSystemDataset(Dataset):
    def __init__(self, data, labels, fault_type=None):
        # Extract features for each waveform
        self.features = np.array([FeatureExtractor.get_features(wave) for wave in data])
        
        if fault_type is not None:
            # Binary classification for specific fault type
            self.labels = torch.FloatTensor(labels == fault_type)
        else:
            self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]

def train_binary_classifier(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                          device, model_path, fault_name, num_epochs=100, patience=10):
    """Training loop for binary classifier."""
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(data).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'\n{fault_name} - Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss / len(val_loader))
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, str(model_path / f'{fault_name}_classifier.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered for {fault_name}")
            break
    
    return best_val_acc

def load_data(data_path):
    """Load and preprocess training and validation data."""
    with h5py.File(data_path, 'r') as f:
        x_train = f['train/data'][:]
        y_train = f['train/labels'][:]
        x_val = f['val/data'][:]
        y_val = f['val/labels'][:]
        fault_types = [x.decode('utf-8') for x in f['fault_types'][:]]
    
    return x_train, y_train, x_val, y_val, fault_types

def main():
    # Setup
    base_dir = Path("ml/training")
    data_path = base_dir / "data/pytorch/fault_classification.h5"
    model_path = base_dir / "models/pytorch"
    model_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    x_train, y_train, x_val, y_val, fault_types = load_data(data_path)
    
    # Extract features to get input size
    n_features = len(FeatureExtractor.get_features(x_train[0]))
    print(f"\nNumber of extracted features: {n_features}")
    
    # Train binary classifier for each fault type
    best_accuracies = {}
    
    for i, fault_type in enumerate(fault_types):
        print(f"\nTraining classifier for: {fault_type}")
        
        # Create datasets for this fault type
        train_dataset = PowerSystemDataset(x_train, y_train, fault_type=i)
        val_dataset = PowerSystemDataset(x_val, y_val, fault_type=i)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model and training components
        model = BinaryClassifier(n_features).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Train the model
        best_acc = train_binary_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_path=model_path,
            fault_name=fault_type
        )
        
        best_accuracies[fault_type] = best_acc
    
    print("\nTraining complete! Best validation accuracies:")
    for fault_type, acc in best_accuracies.items():
        print(f"{fault_type}: {acc:.2f}%")
    
    # Save fault type mapping
    import json
    with open(model_path / 'fault_types.json', 'w') as f:
        json.dump(fault_types, f)

if __name__ == "__main__":
    main()