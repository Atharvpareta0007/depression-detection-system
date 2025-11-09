"""
Evaluate model predictions and compute metrics
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_predictions(predictions_file):
    """
    Load predictions from JSON file
    
    Expected format:
    {
        "predictions": ["depressed", "healthy", ...],
        "labels": ["depressed", "healthy", ...],
        "probabilities": [[0.2, 0.8], [0.9, 0.1], ...]
    }
    """
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    return data


def compute_metrics(predictions, labels, probabilities=None, class_names=None):
    """
    Compute evaluation metrics
    
    Args:
        predictions: List of predicted class labels
        labels: List of true class labels
        probabilities: Optional list of probability vectors
        class_names: Optional list of class names
        
    Returns:
        Dictionary with metrics
    """
    if class_names is None:
        class_names = sorted(set(labels + predictions))
    
    # Convert to numeric if needed
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    y_true = [label_to_idx[label] for label in labels]
    y_pred = [label_to_idx[pred] for pred in predictions]
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (if probabilities provided)
    auc = None
    if probabilities is not None:
        try:
            # Convert probabilities to numpy array
            prob_array = np.array(probabilities)
            if prob_array.shape[1] == 2:
                # Binary classification
                auc = roc_auc_score(y_true, prob_array[:, 1])
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
    
    # Per-class metrics dictionary
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[idx]),
            'recall': float(recall_per_class[idx]),
            'f1': float(f1_per_class[idx])
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    if auc is not None:
        metrics['roc_auc'] = float(auc)
    
    return metrics


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate(predictions_file, labels_file=None, output_dir='reports',
            class_names=None):
    """
    Evaluate predictions and save metrics
    
    Args:
        predictions_file: Path to predictions JSON file
        labels_file: Optional path to labels JSON file (if separate)
        output_dir: Output directory for reports
        class_names: Optional list of class names
    """
    # Load predictions
    pred_data = load_predictions(predictions_file)
    
    predictions = pred_data.get('predictions', [])
    labels = pred_data.get('labels', [])
    probabilities = pred_data.get('probabilities', None)
    
    # Load labels from separate file if provided
    if labels_file:
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        labels = labels_data.get('labels', labels)
    
    if len(predictions) != len(labels):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(labels)} labels")
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, probabilities, class_names)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1:        {class_metrics['f1']:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, metrics['class_names'], cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save classification report
    report = classification_report(
        labels, predictions,
        target_names=metrics['class_names'],
        output_dict=True
    )
    report_file = os.path.join(output_dir, 'classification_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved to: {report_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels JSON file (if separate)')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Class names (e.g., healthy depressed)')
    
    args = parser.parse_args()
    
    # Evaluate
    metrics = evaluate(
        predictions_file=args.predictions,
        labels_file=args.labels,
        output_dir=args.output_dir,
        class_names=args.class_names
    )
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()

