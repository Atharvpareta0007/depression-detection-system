# Temporal Model Architectures

This document describes the alternative model architectures for depression detection from speech.

## Overview

The project supports three model architectures:

1. **CNN** (baseline): Enhanced CNN with batch normalization
2. **CNN-LSTM**: CNN front-end + BiLSTM temporal pooling
3. **Transformer**: Lightweight Transformer encoder with positional encodings

## Model Architectures

### 1. CNN (Baseline)

**Location**: `src/model.py`

- **Architecture**: 3 Conv1D layers + Global Average Pooling + MLP classifier
- **Input**: 120 MFCC features × 31 time frames
- **Output**: 2 classes (Healthy, Depressed)
- **Parameters**: ~6.1M

**Features**:
- Batch normalization
- Progressive dropout (0.6 → 0.4 → 0.3)
- Global average pooling

### 2. CNN-LSTM

**Location**: `src/models/cnn_lstm.py`

- **Architecture**: CNN front-end + BiLSTM + MLP classifier
- **Input**: 120 MFCC features × 31 time frames
- **Output**: 2 classes (Healthy, Depressed)
- **Parameters**: ~8.5M

**Features**:
- CNN feature extraction (3 Conv1D layers)
- Bidirectional LSTM for temporal modeling
- Hidden size: 128, Layers: 2
- Last hidden state pooling

**Advantages**:
- Better temporal modeling
- Captures long-term dependencies
- Bidirectional context

### 3. Transformer

**Location**: `src/models/transformer_model.py`

- **Architecture**: Input projection + Positional encoding + Transformer encoder + MLP classifier
- **Input**: 120 MFCC features × 31 time frames
- **Output**: 2 classes (Healthy, Depressed)
- **Parameters**: ~5.2M

**Features**:
- d_model: 128
- nhead: 8
- num_layers: 3
- dim_feedforward: 512
- Positional encodings
- Mean pooling over time

**Advantages**:
- Self-attention mechanism
- Parallel processing
- Better long-range dependencies

## Training

### Using train_experiments.py

```bash
# Train CNN-LSTM model
python train_experiments.py --model cnn_lstm --data data/training_data.csv --epochs 50

# Train Transformer model
python train_experiments.py --model transformer --data data/training_data.csv --epochs 50

# Train baseline CNN
python train_experiments.py --model cnn --data data/training_data.csv --epochs 50
```

### Arguments

- `--model`: Model type (`cnn`, `cnn_lstm`, `transformer`)
- `--data`: Path to CSV data file with `audio_path` and `label` columns
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.5)
- `--output_dir`: Output directory for checkpoints (default: `experiments`)

### Output

Models are saved to `experiments/{model_type}/`:
- `best_model.pth`: Best model checkpoint
- `history.json`: Training history (loss, accuracy)

## Hyperparameters

### Default Hyperparameters

| Model | Learning Rate | Batch Size | Dropout | Epochs |
|-------|---------------|------------|---------|--------|
| CNN | 0.001 | 32 | 0.6 | 50 |
| CNN-LSTM | 0.001 | 32 | 0.5 | 50 |
| Transformer | 0.001 | 32 | 0.5 | 50 |

### Recommended Hyperparameters

For best performance, consider:
- **Learning rate**: 0.0001 - 0.001 (use learning rate scheduling)
- **Batch size**: 16 - 64 (depending on GPU memory)
- **Dropout**: 0.3 - 0.6 (higher for overfitting)
- **Epochs**: 50 - 100 (with early stopping)

## Evaluation

After training, evaluate models using:

```bash
python scripts/evaluate.py --predictions predictions.json --labels labels.json
```

## Comparison

| Model | Parameters | Training Time | Memory | Accuracy* |
|-------|------------|---------------|--------|-----------|
| CNN | 6.1M | Fast | Low | 75% |
| CNN-LSTM | 8.5M | Medium | Medium | TBD |
| Transformer | 5.2M | Medium | Medium | TBD |

*Accuracy depends on dataset and training configuration

## Usage in Inference

To use a trained model in inference:

```python
from src.models.cnn_lstm import CNNLSTMModel
from src.models.transformer_model import TransformerModel

# Load CNN-LSTM model
model = CNNLSTMModel()
checkpoint = torch.load('experiments/cnn_lstm/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Use in inference
```

## Notes

- All models use the same input format: 120 MFCC features × 31 time frames
- Models are trained with data augmentation (time stretch, pitch shift, noise)
- Early stopping is recommended to prevent overfitting
- Use cross-validation for reliable performance estimates

