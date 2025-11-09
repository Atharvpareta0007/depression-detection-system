"""
Saliency maps and Integrated Gradients for model explainability
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SaliencyExplainer:
    """
    Compute saliency maps using gradients w.r.t. input features
    """
    
    def __init__(self, model, device=None):
        """
        Args:
            model: Trained PyTorch model
            device: Device to run on
        """
        self.model = model
        self.model.eval()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
    
    def compute_saliency(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute saliency map using gradients
        
        Args:
            input_tensor: Input tensor (batch_size, features, time_steps)
            target_class: Target class index (None for predicted class)
            
        Returns:
            Saliency map (features, time_steps)
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients
        saliency = input_tensor.grad.data.abs()
        
        # Average over batch if needed
        if saliency.dim() > 2:
            saliency = saliency.squeeze(0)
        
        return saliency.cpu().numpy()
    
    def compute_integrated_gradients(self, input_tensor: torch.Tensor, 
                                     baseline: Optional[torch.Tensor] = None,
                                     steps: int = 50,
                                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute Integrated Gradients
        
        Args:
            input_tensor: Input tensor
            baseline: Baseline input (zeros if None)
            steps: Number of integration steps
            target_class: Target class index
            
        Returns:
            Integrated gradients (features, time_steps)
        """
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            
            # Get target class
            if target_class is None:
                target_class = torch.argmax(output, dim=1)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
            
            # Get gradients
            grad = interpolated.grad.data
            gradients.append(grad)
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients = (input - baseline) * avg_gradients
        integrated = (input_tensor - baseline) * avg_gradients
        
        # Average over batch if needed
        if integrated.dim() > 2:
            integrated = integrated.squeeze(0)
        
        return integrated.abs().cpu().numpy()
    
    def generate_heatmap(self, saliency_map: np.ndarray, 
                        feature_names: Optional[list] = None) -> str:
        """
        Generate heatmap visualization as base64 PNG
        
        Args:
            saliency_map: Saliency map (features, time_steps)
            feature_names: Optional feature names
            
        Returns:
            Base64-encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(saliency_map, aspect='auto', cmap='hot', interpolation='nearest')
        
        # Labels
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Saliency Map (Feature Importance)', fontsize=14, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Importance')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def explain(self, input_tensor: torch.Tensor, 
                method: str = 'saliency',
                target_class: Optional[int] = None) -> Dict:
        """
        Generate explanation for input
        
        Args:
            input_tensor: Input tensor
            method: 'saliency' or 'integrated_gradients'
            target_class: Target class index
            
        Returns:
            Dictionary with explanation results
        """
        if method == 'saliency':
            importance_map = self.compute_saliency(input_tensor, target_class)
        elif method == 'integrated_gradients':
            importance_map = self.compute_integrated_gradients(input_tensor, target_class=target_class)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Generate heatmap
        heatmap_base64 = self.generate_heatmap(importance_map)
        
        # Compute per-time-frame importance
        per_time_importance = importance_map.mean(axis=0).tolist()
        
        # Compute per-feature importance
        per_feature_importance = importance_map.mean(axis=1).tolist()
        
        # Top features
        top_features = []
        if len(per_feature_importance) > 0:
            top_indices = np.argsort(per_feature_importance)[-5:][::-1]
            for idx in top_indices:
                feature_name = f"feature_{idx}" if len(per_feature_importance) <= 120 else f"mfcc_{idx}"
                top_features.append([feature_name, float(per_feature_importance[idx])])
        
        return {
            'explanation_type': method,
            'heatmap': heatmap_base64,
            'per_time_importance': per_time_importance,
            'per_feature_importance': per_feature_importance,
            'top_features': top_features
        }


def explain_audio(model, audio_features: np.ndarray, 
                 method: str = 'saliency',
                 device: Optional[torch.device] = None) -> Dict:
    """
    Explain audio prediction using saliency maps
    
    Args:
        model: Trained PyTorch model
        audio_features: Audio features (features, time_steps) or (batch, features, time_steps)
        method: 'saliency' or 'integrated_gradients'
        device: Device to run on
        
    Returns:
        Dictionary with explanation results
    """
    explainer = SaliencyExplainer(model, device=device)
    
    # Convert to tensor
    if isinstance(audio_features, np.ndarray):
        if audio_features.ndim == 2:
            audio_features = torch.FloatTensor(audio_features).unsqueeze(0)
        else:
            audio_features = torch.FloatTensor(audio_features)
    
    # Generate explanation
    explanation = explainer.explain(audio_features, method=method)
    
    return explanation

