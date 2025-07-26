"""
Script to save the trained model and class names from the training environment.
Run this in your training environment after training is complete.
"""
import os
import json
import torch
from torch import nn
import timm

# Assuming this is run in your training environment
# where you have access to label_encoder and the trained model

# 1. Save class names
os.makedirs('models', exist_ok=True)
with open('models/classes.json', 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)

print(f"Saved class names to models/classes.json: {label_encoder.classes_.tolist()}")

# 2. Save the model
model_path = 'models/best_knee_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Saved model weights to {model_path}")

# 3. Verify model can be loaded
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = timm.create_model('resnet18', pretrained=False, num_classes=len(label_encoder.classes_), in_chans=1)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.eval()
print("Model loaded and verified successfully!")
