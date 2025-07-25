import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for model explainability.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The trained model.
            target_layer: The target layer to compute Grad-CAM for.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]
        
        def forward_hook(module, input, output):
            self.activation = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def _get_gradcam_heatmap(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input and target class.
        
        Args:
            input_tensor: Input tensor (batch size 1).
            target_class: Target class index.
            
        Returns:
            Heatmap as a numpy array.
        """
        # Forward pass
        model_output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(model_output)
        one_hot[0][target_class] = 1.0
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradient.detach().cpu().numpy()[0]
        activations = self.activation.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2), keepdims=True)
        
        # Weighted combination of activation maps
        cam = np.sum(weights * activations, axis=0)
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Avoid division by zero
        
        return cam
    
    def visualize(
        self, 
        input_tensor: torch.Tensor, 
        original_image: np.ndarray,
        target_class: int,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, str]:
        """
        Generate and visualize Grad-CAM heatmap overlaid on the original image.
        
        Args:
            input_tensor: Input tensor (batch size 1).
            original_image: Original image as a numpy array (H, W, C).
            target_class: Target class index.
            alpha: Opacity for the heatmap overlay.
            colormap: OpenCV colormap to use.
            
        Returns:
            Tuple of (overlayed image, base64 encoded image)
        """
        # Generate heatmap
        heatmap = self._get_gradcam_heatmap(input_tensor, target_class)
        
        # Convert to 8-bit
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Resize heatmap to match original image size
        heatmap_colored = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
        
        # Convert original image to BGR if grayscale
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Ensure both images have the same data type
        original_image = original_image.astype(np.float32) / 255.0
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        overlayed = np.clip(overlayed * 255, 0, 255).astype(np.uint8)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        return overlayed, base64_img

def get_gradcam_visualization(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: int,
    target_layer: Optional[torch.nn.Module] = None
) -> Tuple[np.ndarray, str]:
    """
    Helper function to get Grad-CAM visualization.
    
    Args:
        model: The trained model.
        input_tensor: Input tensor (batch size 1).
        original_image: Original image as a numpy array (H, W, C).
        target_class: Target class index.
        target_layer: Target layer for Grad-CAM. If None, will use the last conv layer.
        
    Returns:
        Tuple of (overlayed image, base64 encoded image)
    """
    # If target layer is not specified, find the last convolutional layer
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError("No convolutional layer found in the model")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    
    # Get visualization
    return grad_cam.visualize(
        input_tensor=input_tensor,
        original_image=original_image,
        target_class=target_class
    )
