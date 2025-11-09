# Evaluation Protocol

This document describes the evaluation protocol for the depression detection system.

## Overview

The evaluation system computes comprehensive metrics including:
- Accuracy
- Precision (per-class and weighted)
- Recall (per-class and weighted)
- F1-Score (per-class and weighted)
- ROC AUC (if probabilities provided)
- Confusion Matrix

## Data Split

### Stratified Train/Val/Test Split

Use `scripts/split_dataset.py` to create stratified splits:

```bash
python scripts/split_dataset.py \
    --data data/training_data.csv \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --speaker_column speaker_id  # Optional: for speaker-level separation
```

### Split Ratios

- **Training**: 70% (default)
- **Validation**: 15% (default)
- **Test**: 15% (default)

### Speaker-Level Separation

If speaker metadata is available, use `--speaker_column` to ensure:
- No speaker appears in multiple splits
- Better generalization to unseen speakers

## Evaluation Script

### Using evaluate.py

```bash
python scripts/evaluate.py \
    --predictions predictions.json \
    --labels labels.json \
    --output_dir reports \
    --class_names healthy depressed
```

### Input Format

**predictions.json**:
```json
{
    "predictions": ["depressed", "healthy", "depressed", ...],
    "labels": ["depressed", "healthy", "healthy", ...],
    "probabilities": [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], ...]
}
```

**labels.json** (if separate):
```json
{
    "labels": ["depressed", "healthy", "healthy", ...]
}
```

### Output

The script generates:
- `reports/metrics.json`: Comprehensive metrics
- `reports/confusion_matrix.png`: Confusion matrix visualization
- `reports/classification_report.json`: Detailed classification report

## Metrics

### Overall Metrics

- **Accuracy**: Overall classification accuracy
- **Precision** (weighted): Average precision across classes
- **Recall** (weighted): Average recall across classes
- **F1-Score** (weighted): Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (if probabilities provided)

### Per-Class Metrics

For each class (Healthy, Depressed):
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Confusion Matrix

| | Predicted Healthy | Predicted Depressed |
|--|-------------------|---------------------|
| **Actual Healthy** | TN | FP |
| **Actual Depressed** | FN | TP |

Where:
- **TN**: True Negatives
- **FP**: False Positives
- **FN**: False Negatives
- **TP**: True Positives

## Metrics Interpretation

### Accuracy

- **Range**: 0.0 - 1.0
- **Interpretation**: Overall proportion of correct predictions
- **Note**: Can be misleading with imbalanced classes

### Precision

- **Range**: 0.0 - 1.0
- **Interpretation**: Of all positive predictions, how many are correct?
- **High precision**: Few false positives

### Recall

- **Range**: 0.0 - 1.0
- **Interpretation**: Of all actual positives, how many are detected?
- **High recall**: Few false negatives

### F1-Score

- **Range**: 0.0 - 1.0
- **Interpretation**: Balanced metric combining precision and recall
- **Use case**: When both precision and recall are important

### ROC AUC

- **Range**: 0.0 - 1.0
- **Interpretation**: Ability to distinguish between classes
- **0.5**: Random classifier
- **1.0**: Perfect classifier

## Evaluation Best Practices

### 1. Use Test Set Only for Final Evaluation

- Train on training set
- Tune hyperparameters on validation set
- Evaluate final model on test set (only once)

### 2. Report Multiple Metrics

- Don't rely on accuracy alone
- Report precision, recall, F1, and AUC
- Include per-class metrics

### 3. Handle Class Imbalance

- Use stratified splits
- Report per-class metrics
- Consider class weights in training

### 4. Cross-Validation

For small datasets, use k-fold cross-validation:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Train and evaluate
```

### 5. Statistical Significance

- Report confidence intervals
- Use statistical tests for comparison
- Multiple runs for stability

## Example Output

```json
{
    "accuracy": 0.75,
    "precision": 0.74,
    "recall": 0.76,
    "f1_score": 0.75,
    "roc_auc": 0.82,
    "per_class": {
        "Healthy": {
            "precision": 0.73,
            "recall": 0.78,
            "f1": 0.75
        },
        "Depressed": {
            "precision": 0.75,
            "recall": 0.74,
            "f1": 0.75
        }
    },
    "confusion_matrix": [[39, 11], [10, 40]]
}
```

## Integration with Training

### Save Best Model by F1

Modify training script to save best model by F1-score:

```python
from sklearn.metrics import f1_score

val_f1 = f1_score(y_true, y_pred, average='weighted')
if val_f1 > best_f1:
    torch.save(model.state_dict(), 'best_model.pth')
    best_f1 = val_f1
```

### Log Metrics During Training

Save metrics to JSON for analysis:

```python
metrics = {
    'epoch': epoch,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
    'val_f1': val_f1
}
with open('training_metrics.json', 'a') as f:
    json.dump(metrics, f)
```

## Troubleshooting

### Low Accuracy

- Check data quality
- Verify preprocessing
- Try different hyperparameters
- Consider data augmentation

### High Precision, Low Recall

- Model is conservative (few false positives, many false negatives)
- Lower classification threshold
- Adjust class weights

### Low Precision, High Recall

- Model is aggressive (many false positives, few false negatives)
- Raise classification threshold
- Adjust class weights

## References

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix Guide](https://en.wikipedia.org/wiki/Confusion_matrix)
- [ROC AUC Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

