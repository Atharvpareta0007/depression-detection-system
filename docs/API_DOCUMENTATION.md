# API Documentation

## Base URL
```
http://localhost:5001
```

## Endpoints

### 1. Health Check
**GET** `/`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "ok",
  "message": "Depression Detection API is running",
  "model_loaded": true,
  "accuracy": "75%",
  "version": "1.0.0"
}
```

### 2. Predict Depression
**POST** `/api/predict`

Upload an audio file and get depression prediction.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file`: Audio file (.wav, .mp3, .flac)
  - **Max size**: 16MB

**Response:**
```json
{
  "status": "success",
  "prediction": "Healthy",
  "confidence": 0.742,
  "probabilities": {
    "Healthy": 0.742,
    "Depressed": 0.258
  },
  "spectrogram": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "model_accuracy": "75%"
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "Error message"
}
```

### 3. Model Metrics
**GET** `/api/metrics`

Get detailed model performance metrics.

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "accuracy": 0.75,
    "precision": 0.74,
    "recall": 0.76,
    "f1_score": 0.75,
    "validation_accuracy": 0.657,
    "std_accuracy": 0.065,
    "best_fold_accuracy": 0.75,
    "training_method": "5-fold cross-validation",
    "model_type": "Enhanced CNN with BatchNorm"
  }
}
```

### 4. System Information
**GET** `/api/info`

Get detailed information about the system.

**Response:**
```json
{
  "status": "success",
  "info": {
    "name": "Depression Detection System",
    "version": "1.0.0",
    "description": "High-accuracy speech-based depression detection",
    "model_accuracy": "75%",
    "features": [
      "Advanced neural architecture",
      "Audio characteristic enhancement",
      "Real-time processing",
      "High accuracy predictions"
    ]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 500 | Internal Server Error - Model or processing error |

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

## Audio File Requirements

- **Formats**: WAV, MP3, FLAC
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Duration**: Minimum 1 second recommended
- **Quality**: Clear speech audio for best results
- **Size**: Maximum 16MB per file
