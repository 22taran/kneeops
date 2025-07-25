from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import io
import json
import pickle
import os
import base64
from typing import Dict, Any, List, Tuple, Optional
import timm

# Import Grad-CAM utility
from grad_cam import get_gradcam_visualization

# Simple CNN model for knee injury classification
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

# Initialize FastAPI app
app = FastAPI(title="KneeOps API", description="API for knee MRI classification")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
transform = None
# Load class names from file if available
try:
    if os.path.exists('models/classes.json'):
        with open('models/classes.json', 'r') as f:
            class_names = json.load(f)
    elif os.path.exists('models/classes.npy'):
        class_names = np.load('models/classes.npy').tolist()
    else:
        class_names = ["Healthy", "ACL Injury", "Meniscus Tear"]
except Exception as e:
    print(f"Warning: Could not load class names from file: {e}")
    class_names = ["Healthy", "ACL Injury", "Meniscus Tear"]
label_map = {i: name for i, name in enumerate(class_names)}

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

def load_model():
    """Load or create the model"""
    global model, device, transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model (ResNet18 from timm, matching training)
    model = timm.create_model('resnet18', pretrained=False, num_classes=len(label_map), in_chans=1)
    model.to(device)
    model.eval()
    
    # Try to load trained weights
    model_path = 'models/best_knee_model.pth'
    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("✅ Loaded model with 'state_dict' format")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Loaded model with 'model_state_dict' format")
                else:
                    model.load_state_dict(checkpoint)
                    print("✅ Loaded model with direct state_dict format")
            elif hasattr(checkpoint, 'state_dict'):
                model.load_state_dict(checkpoint.state_dict())
                print("✅ Loaded model from entire model object")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded model with direct state_dict format")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using randomly initialized model (confidence will be ~33% for 3 classes)")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("⚠️  Using randomly initialized model (confidence will be ~33% for 3 classes)")
    
    # Set up transform
    transform = get_transform()
    
    print("Model initialized successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "KneeOps API is running"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": "2025-07-12T02:53:51.559626"
    }

@app.get("/api/model-info")
async def get_model_info():
    return {
        "model_info": {
            "device": str(device),
            "label_map": label_map,
            "model_path": "models/best_model.pt",
            "num_classes": len(label_map)
        },
        "success": True
    }

def process_image_data(image_data, return_original=False):
    """Process image data from .pck file
    
    Args:
        image_data: Input image data (numpy array or tensor)
        return_original: If True, returns both the processed tensor and original image
        
    Returns:
        Processed tensor (and optionally the original image)
    """
    try:
        # Store original for visualization
        original_image = None
        
        # Convert to numpy array if needed
        if isinstance(image_data, torch.Tensor):
            original_image = image_data.numpy()
        else:
            original_image = np.array(image_data)
        
        if not isinstance(original_image, np.ndarray):
            raise ValueError("Image data must be a numpy array or tensor")
        
        # Ensure we have a 2D image for processing
        if len(original_image.shape) == 3 and original_image.shape[2] == 1:
            # Single channel image
            original_image = original_image[:, :, 0]
        elif len(original_image.shape) == 3 and original_image.shape[2] > 3:
            # Multi-channel image, take first channel
            original_image = original_image[:, :, 0]
        
        # Create a copy for processing
        processed_image = original_image.copy()
        
        # Normalize to 0-255 range if needed
        if processed_image.dtype != np.uint8:
            min_val = processed_image.min()
            max_val = processed_image.max()
            if max_val > min_val:  # Avoid division by zero
                processed_image = ((processed_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                processed_image = np.zeros_like(processed_image, dtype=np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed_image, mode='L')
        
        # Apply transforms
        image_tensor = transform(pil_image)
        
        if return_original:
            # Convert original to RGB if needed for visualization
            if len(original_image.shape) == 2:  # Grayscale
                original_image = np.stack((original_image,) * 3, axis=-1)
            return image_tensor, original_image
            
        return image_tensor
        
    except Exception as e:
        print(f"Error processing image data: {str(e)}")
        return None

@app.post("/api/upload-mri")
async def upload_mri(file: UploadFile = File(...)):
    """Upload and analyze MRI file"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith('.pck'):
        raise HTTPException(status_code=400, detail="File must be a .pck file")
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Load pickle data
        try:
            pck_data = pickle.loads(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid .pck file: {str(e)}")
        
        # Process images
        processed_tensors = []
        
        if isinstance(pck_data, list):
            for item in pck_data:
                tensor = process_image_data(item)
                if tensor is not None:
                    processed_tensors.append(tensor)
        else:
            tensor = process_image_data(pck_data)
            if tensor is not None:
                processed_tensors.append(tensor)
        
        if not processed_tensors:
            raise HTTPException(status_code=400, detail="No valid image data found in .pck file")
        
        # Make predictions
        all_predictions = []
        overall_analysis = {
            "total_images": len(processed_tensors),
            "processed_images": 0,
            "average_confidence": 0.0,
            "most_common_prediction": None,
            "prediction_distribution": {},
            "severity_analysis": {
                "healthy_count": 0,
                "injury_count": 0,
                "severe_injury_count": 0
            }
        }
        
        for i, image_tensor in enumerate(processed_tensors):
            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Debug information
            print(f"🔍 Image {i} Debug Info:")
            print(f"   Raw outputs: {outputs[0].tolist()}")
            print(f"   Probabilities: {probabilities[0].tolist()}")
            print(f"   Predicted class: {predicted_class} ({label_map[predicted_class]})")
            print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Get all class probabilities
            all_probabilities = {}
            for class_idx, class_name in label_map.items():
                all_probabilities[class_name] = float(probabilities[0][class_idx].item())
            
            # Determine severity
            severity = "healthy"
            if predicted_class == 1:  # ACL Injury
                severity = "mild_injury"
            elif predicted_class == 2:  # Meniscus Tear
                severity = "severe_injury"
            
            prediction_result = {
                "image_index": i,
                "class": label_map[predicted_class],
                "class_index": predicted_class,
                "confidence": float(confidence),
                "confidence_percentage": float(confidence * 100),
                "all_probabilities": all_probabilities,
                "severity": severity,
                "recommendations": _get_recommendations(label_map[predicted_class], confidence)
            }
            all_predictions.append(prediction_result)
            
            # Update overall analysis
            overall_analysis["processed_images"] += 1
            overall_analysis["average_confidence"] += confidence
            
            # Update prediction distribution
            class_name = label_map[predicted_class]
            overall_analysis["prediction_distribution"][class_name] = overall_analysis["prediction_distribution"].get(class_name, 0) + 1
            
            # Update severity counts
            if predicted_class == 0:
                overall_analysis["severity_analysis"]["healthy_count"] += 1
            elif predicted_class == 1:
                overall_analysis["severity_analysis"]["injury_count"] += 1
            else:
                overall_analysis["severity_analysis"]["severe_injury_count"] += 1
        
        # Calculate final statistics
        if overall_analysis["processed_images"] > 0:
            overall_analysis["average_confidence"] /= overall_analysis["processed_images"]
            
            # Find most common prediction
            if overall_analysis["prediction_distribution"]:
                most_common = max(overall_analysis["prediction_distribution"], key=overall_analysis["prediction_distribution"].get)
                overall_analysis["most_common_prediction"] = most_common
        
        # Determine overall severity
        overall_analysis["overall_severity"] = _determine_overall_severity(overall_analysis)
        overall_analysis["overall_recommendations"] = _get_overall_recommendations(overall_analysis)
        
        # Prepare MRI data for response
        mri_data_for_response = []
        for tensor in processed_tensors:
            # Convert tensor to numpy and normalize for display
            img_data = tensor.cpu().numpy()
            # If it's a batch, take first item
            if len(img_data.shape) == 4:
                img_data = img_data[0]
            # If it's a single channel image, remove the channel dimension for compatibility
            if len(img_data.shape) == 3 and img_data.shape[0] == 1:
                img_data = img_data[0]
            mri_data_for_response.append(img_data.tolist())
        
        # Return response with MRI data
        if len(all_predictions) == 1:
            response = {
                "success": True,
                "prediction": all_predictions[0],
                "filename": file.filename,
                "total_images": len(processed_tensors),
                "overall_analysis": overall_analysis,
                "mri_data": mri_data_for_response[0] if mri_data_for_response else None
            }
        else:
            response = {
                "success": True,
                "predictions": all_predictions,
                "filename": file.filename,
                "total_images": len(processed_tensors),
                "overall_analysis": overall_analysis,
                "mri_data": mri_data_for_response
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/chat")
async def chat(message: str, file_id: str = None):
    """Simple chat endpoint"""
    return {
        "success": True,
        "response": f"Based on the MRI analysis, I can help answer your question: '{message}'. For detailed medical advice, please consult with a healthcare professional.",
        "analysis_context": {
            "acl_status": "unknown",
            "severity": "unknown",
            "confidence": 0.0
        }
    }

def _get_recommendations(predicted_class: str, confidence: float) -> list:
    """Generate recommendations based on prediction and confidence"""
    recommendations = []
    
    if predicted_class == "Healthy":
        if confidence > 0.8:
            recommendations.append("No significant knee injury detected")
            recommendations.append("Continue with normal activities")
        else:
            recommendations.append("Consider follow-up imaging for confirmation")
    elif predicted_class == "ACL Injury":
        recommendations.append("ACL injury detected - consult orthopedic specialist")
        recommendations.append("Consider MRI for detailed assessment")
        if confidence > 0.7:
            recommendations.append("High confidence in ACL injury diagnosis")
        else:
            recommendations.append("Moderate confidence - additional imaging recommended")
    elif predicted_class == "Meniscus Tear":
        recommendations.append("Meniscus tear detected - urgent orthopedic consultation")
        recommendations.append("Consider surgical evaluation")
        if confidence > 0.7:
            recommendations.append("High confidence in meniscus tear diagnosis")
        else:
            recommendations.append("Moderate confidence - additional imaging recommended")
    
    recommendations.append("This is an AI analysis - consult healthcare professional for final diagnosis")
    return recommendations

def _determine_overall_severity(analysis: dict) -> str:
    """Determine overall severity based on all predictions"""
    severity_counts = analysis["severity_analysis"]
    
    if severity_counts["severe_injury_count"] > 0:
        return "severe_injury"
    elif severity_counts["injury_count"] > 0:
        return "mild_injury"
    else:
        return "healthy"

def _get_overall_recommendations(analysis: dict) -> list:
    """Generate overall recommendations based on all predictions"""
    recommendations = []
    
    total_images = analysis["total_images"]
    avg_confidence = analysis["average_confidence"]
    
    recommendations.append(f"Analyzed {total_images} MRI images")
    recommendations.append(f"Average confidence: {avg_confidence:.1%}")
    
    if analysis["overall_severity"] == "severe_injury":
        recommendations.append("⚠️ Severe injury detected - Urgent medical attention required")
        recommendations.append("Immediate orthopedic consultation recommended")
    elif analysis["overall_severity"] == "mild_injury":
        recommendations.append("⚠️ Injury detected - Medical consultation recommended")
        recommendations.append("Schedule appointment with orthopedic specialist")
    else:
        recommendations.append("✅ No significant injuries detected")
        recommendations.append("Continue with normal activities")
    
    recommendations.append("This analysis is for screening purposes only")
    recommendations.append("Final diagnosis should be made by healthcare professional")
    
    return recommendations

@app.post("/api/visualize-gradcam")
async def visualize_gradcam(
    file: UploadFile = File(...),
    slice_index: int = Query(0, description="Slice index for 3D MRI volumes"),
    target_class: int = Query(..., description="Target class index for Grad-CAM")
):
    """Generate Grad-CAM visualization for the uploaded MRI"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.pck'):
        raise HTTPException(status_code=400, detail="File must be a .pck file")
    
    try:
        # Read and process the file
        file_bytes = await file.read()
        pck_data = pickle.loads(file_bytes)
        
        # Handle 3D volumes
        if isinstance(pck_data, np.ndarray) and len(pck_data.shape) == 3:
            if slice_index < 0 or slice_index >= pck_data.shape[2]:
                raise HTTPException(status_code=400, detail=f"Invalid slice index. Must be between 0 and {pck_data.shape[2]-1}")
            image_data = pck_data[:, :, slice_index]
        else:
            image_data = pck_data
        
        # Process image and get both tensor and original
        input_tensor, original_image = process_image_data(image_data, return_original=True)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Get the last convolutional layer for Grad-CAM
        target_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            raise HTTPException(status_code=500, detail="No convolutional layer found in the model")
        
        # Get Grad-CAM visualization
        _, base64_img = get_gradcam_visualization(
            model=model,
            input_tensor=input_tensor,
            original_image=original_image,
            target_class=target_class,
            target_layer=target_layer
        )
        
        return {
            "success": True,
            "visualization": f"data:image/png;base64,{base64_img}",
            "target_class": label_map.get(target_class, f"Class {target_class}")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM visualization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 