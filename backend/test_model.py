#!/usr/bin/env python3
"""
Test script to debug model loading and confidence calculation
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Same model definition as in app.py
class KneeInjuryCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(KneeInjuryCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 80 * 80, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def test_model():
    """Test model loading and prediction"""
    
    device = torch.device("cpu")
    label_map = {0: "Healthy", 1: "ACL Injury", 2: "Meniscus Tear"}
    
    print("🔍 Testing Model Loading and Prediction")
    print("=" * 50)
    
    # Create model
    model = KneeInjuryCNN(num_classes=len(label_map))
    model.to(device)
    model.eval()
    
    # Test with random weights (should give ~33% confidence)
    print("\n📊 Testing with random weights:")
    test_input = torch.randn(1, 1, 320, 320)
    with torch.no_grad():
        outputs = model(test_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        print(f"Raw outputs: {outputs[0].tolist()}")
        print(f"Probabilities: {probabilities[0].tolist()}")
        print(f"Predicted class: {predicted_class} ({label_map[predicted_class]})")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    # Try to load trained model
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        print(f"\n📦 Loading trained model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check checkpoint structure
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try different loading methods
            try:
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("✅ Loaded with 'state_dict' format")
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Loaded with 'model_state_dict' format")
                elif hasattr(checkpoint, 'state_dict'):
                    # The checkpoint is the entire model object
                    model.load_state_dict(checkpoint.state_dict())
                    print("✅ Loaded from entire model object")
                else:
                    model.load_state_dict(checkpoint)
                    print("✅ Loaded with direct state_dict format")
                
                # Test with trained weights
                print("\n📊 Testing with trained weights:")
                with torch.no_grad():
                    outputs = model(test_input)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    print(f"Raw outputs: {outputs[0].tolist()}")
                    print(f"Probabilities: {probabilities[0].tolist()}")
                    print(f"Predicted class: {predicted_class} ({label_map[predicted_class]})")
                    print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error loading state dict: {e}")
                
        except Exception as e:
            print(f"❌ Error loading model file: {e}")
    else:
        print(f"❌ Model file not found: {model_path}")

if __name__ == "__main__":
    test_model() 