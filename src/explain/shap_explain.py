"""
SHAP explainability for depression detection model
"""

import numpy as np
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def explain_with_shap(model, audio_features: np.ndarray,
                     background_data: Optional[np.ndarray] = None,
                     device: Optional[torch.device] = None) -> Dict:
    """
    Compute SHAP values for audio features
    
    Args:
        model: Trained PyTorch model
        audio_features: Audio features (features, time_steps) or flattened (features * time_steps,)
        background_data: Background data for SHAP (optional)
        device: Device to run on
        
    Returns:
        Dictionary with SHAP explanation results
    """
    if not SHAP_AVAILABLE:
        return {
            'error': 'SHAP not available. Install with: pip install shap',
            'top_features': []
        }
    
    model.eval()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # Flatten features if needed
    if audio_features.ndim == 2:
        # Average over time or flatten
        # For MFCC features, we'll use mean over time
        features_flat = audio_features.mean(axis=1)  # (features,)
    else:
        features_flat = audio_features.flatten()
    
    # Create background data if not provided
    if background_data is None:
        # Use zeros as baseline
        background_data = np.zeros((1, len(features_flat)))
    else:
        if background_data.ndim == 2:
            background_data = background_data.mean(axis=1)
        background_data = background_data.reshape(1, -1)
    
    # Create SHAP explainer
    def model_wrapper(x):
        """Wrapper function for SHAP"""
        x_tensor = torch.FloatTensor(x).to(device)
        # Reshape to model input format
        # Assuming model expects (batch, features, time_steps)
        # For now, we'll use mean features
        batch_size = x_tensor.shape[0]
        features = x_tensor.shape[1]
        
        # Create dummy time dimension
        x_reshaped = x_tensor.unsqueeze(-1).expand(batch_size, features, 31)
        
        with torch.no_grad():
            output = model(x_reshaped)
            probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()
    
    try:
        explainer = shap.Explainer(model_wrapper, background_data)
        shap_values = explainer(features_flat.reshape(1, -1))
        
        # Get SHAP values for positive class (depressed)
        shap_values_array = shap_values.values[0, :, 1]  # Assuming class 1 is depressed
        
        # Get top features
        top_indices = np.argsort(np.abs(shap_values_array))[-5:][::-1]
        top_features = []
        
        for idx in top_indices:
            feature_name = f"mfcc_{idx}" if len(shap_values_array) <= 120 else f"feature_{idx}"
            top_features.append([feature_name, float(shap_values_array[idx])])
        
        return {
            'explanation_type': 'shap',
            'shap_values': shap_values_array.tolist(),
            'top_features': top_features,
            'base_value': float(shap_values.base_values[0, 1])
        }
    
    except Exception as e:
        return {
            'error': f'SHAP computation failed: {str(e)}',
            'top_features': []
        }


def explain_handcrafted_features(features: np.ndarray, 
                                 feature_names: Optional[List[str]] = None) -> Dict:
    """
    Explain using handcrafted feature importance
    
    Args:
        features: Feature vector (n_features,)
        feature_names: Optional feature names
        
    Returns:
        Dictionary with feature importance
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(features))]
    
    # Use absolute values as importance
    importance = np.abs(features)
    
    # Get top features
    top_indices = np.argsort(importance)[-5:][::-1]
    top_features = []
    
    for idx in top_indices:
        top_features.append([feature_names[idx], float(importance[idx])])
    
    return {
        'explanation_type': 'handcrafted',
        'top_features': top_features,
        'feature_importance': importance.tolist()
    }

