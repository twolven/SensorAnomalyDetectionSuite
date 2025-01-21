"""
Validation script for the trained waveform pattern detector.
Visualizes detections and calculates performance metrics.
"""

import os
import json
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from train_waveform_detector import WaveformYOLO, load_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """Load the trained model."""
    model = WaveformYOLO(n_fault_types=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def decode_predictions(predictions, threshold=0.5):
    """Convert model predictions to fault detections."""
    # predictions shape: [batch_size, grid_size, 5 + n_fault_types]
    objectness = torch.sigmoid(predictions[..., 0])
    locations = predictions[..., 1:5]  # x_center, y_center, width, magnitude
    fault_types = torch.sigmoid(predictions[..., 5:])
    
    detections = []
    for i in range(len(predictions)):
        sample_dets = []
        # Find cells with high objectness
        detected_cells = torch.where(objectness[i] > threshold)[0]
        
        for cell_idx in detected_cells:
            loc = locations[i, cell_idx]
            fault_type_idx = torch.argmax(fault_types[i, cell_idx]).item()
            confidence = objectness[i, cell_idx].item()
            
            sample_dets.append({
                'start': loc[0].item(),
                'width': loc[2].item(),
                'magnitude': loc[3].item(),
                'type_idx': fault_type_idx,
                'confidence': confidence
            })
        detections.append(sample_dets)
    
    return detections

def calculate_iou(pred_box, true_box):
    """Calculate IoU between predicted and true fault regions."""
    pred_start = pred_box['start']
    pred_end = pred_start + pred_box['width']
    true_start = true_box['start']
    true_end = true_start + true_box['width']
    
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (true_end - true_start) - intersection
    
    return intersection / union

def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    """Calculate detection and classification metrics."""
    total_detections = 0
    correct_detections = 0
    correct_classifications = 0
    false_positives = 0
    
    for pred_dets, true_dets in zip(predictions, ground_truth):
        total_detections += len(true_dets)
        matched_true = set()
        
        for pred in pred_dets:
            matched = False
            for i, true in enumerate(true_dets):
                if i in matched_true:
                    continue
                    
                iou = calculate_iou(pred, true)
                if iou >= iou_threshold:
                    matched = True
                    matched_true.add(i)
                    correct_detections += 1
                    if pred['type_idx'] == true['type_idx']:
                        correct_classifications += 1
                    break
            
            if not matched:
                false_positives += 1
    
    detection_recall = correct_detections / max(total_detections, 1)
    detection_precision = correct_detections / max(correct_detections + false_positives, 1)
    classification_accuracy = correct_classifications / max(correct_detections, 1)
    
    return {
        'detection_recall': detection_recall,
        'detection_precision': detection_precision,
        'classification_accuracy': classification_accuracy,
        'false_positive_rate': false_positives / max(total_detections, 1)
    }

def plot_detections(waveform, predictions, ground_truth, fault_types, save_path):
    """Visualize waveform with predicted and true fault regions."""
    plt.figure(figsize=(15, 8))
    
    # Plot waveform
    plt.plot(waveform, 'b-', label='Waveform', alpha=0.7)
    
    # Plot ground truth faults
    for fault in ground_truth:
        start_idx = int(fault['start'] * len(waveform))
        width = int(fault['width'] * len(waveform))
        end_idx = min(start_idx + width, len(waveform))
        plt.axvspan(start_idx, end_idx, color='g', alpha=0.2, 
                   label=f'True {fault_types[fault["type_idx"]]}')
    
    # Plot predicted faults
    for pred in predictions:
        start_idx = int(pred['start'] * len(waveform))
        width = int(pred['width'] * len(waveform))
        end_idx = min(start_idx + width, len(waveform))
        plt.axvspan(start_idx, end_idx, color='r', alpha=0.2,
                   label=f'Pred {fault_types[pred["type_idx"]]} ({pred["confidence"]:.2f})')
    
    plt.title('Fault Detection Results')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup paths
    base_dir = Path("ml/training")
    data_path = base_dir / "data/yolo/pattern_detection.h5"
    model_path = base_dir / "models/yolo/waveform_detector.pth"
    output_dir = base_dir / "validation"
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(model_path, device)
    
    # Load validation data
    logger.info("Loading validation data...")
    _, _, x_val, ann_val = load_data(data_path)
    fault_types = ['sag', 'swell', 'interruption', 'harmonic']
    
    # Get predictions
    logger.info("Running predictions...")
    all_predictions = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            batch = torch.FloatTensor(x_val[i:i+batch_size]).to(device)
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)
            outputs = model(batch)
            predictions = decode_predictions(outputs)
            all_predictions.extend(predictions)
    
    # Parse ground truth
    ground_truth = []
    for ann_str in ann_val:
        faults = []
        ann = json.loads(ann_str)
        for fault in ann:
            faults.append({
                'start': fault['start'],
                'width': fault['width'],
                'magnitude': fault['magnitude'],
                'type_idx': fault_types.index(fault['type'])
            })
        ground_truth.append(faults)
    
    # Evaluate performance
    logger.info("Calculating metrics...")
    metrics = evaluate_predictions(all_predictions, ground_truth)
    
    logger.info("\nDetection Performance:")
    logger.info(f"Detection Recall: {metrics['detection_recall']:.3f}")
    logger.info(f"Detection Precision: {metrics['detection_precision']:.3f}")
    logger.info(f"Classification Accuracy: {metrics['classification_accuracy']:.3f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
    
    # Plot some example detections
    logger.info("\nGenerating visualization plots...")
    n_examples = 5
    for i in range(n_examples):
        plot_detections(
            x_val[i],
            all_predictions[i],
            ground_truth[i],
            fault_types,
            output_dir / f'detection_example_{i+1}.png'
        )
    
    logger.info(f"Validation plots saved to {output_dir}")
    
    # Save metrics
    metrics_path = output_dir / 'validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()