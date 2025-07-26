# KneeOps: MRI Knee Injury Classification Web Application

## Overview
KneeOps is a web-based application for classifying knee MRI images to detect injuries using deep learning. It features a React frontend for user interaction and a Python backend (FastAPI) for model inference. The backend uses a ResNet18 convolutional neural network, trained on labeled MRI data, to provide accurate predictions.

---

## Table of Contents
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Model Choice and Rationale](#model-choice-and-rationale)
- [Data Preprocessing](#data-preprocessing)
- [Training Pipeline](#training-pipeline)
- [Backend (FastAPI)](#backend-fastapi)
- [Frontend (React)](#frontend-react)
- [End-to-End Flow](#end-to-end-flow)
- [Why This Approach is Better](#why-this-approach-is-better)
- [Deployment & Maintenance](#deployment--maintenance)
- [References](#references)
- [Future Improvements](#future-improvements)
- [How to Use KneeOps](#how-to-use-kneeops)

---

## Project Structure

```
KneeOps/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main FastAPI application
│   ├── requirements.txt        # Python dependencies
│   └── models/
│       └── model_training.py  # Model training script
│
├── src/                       # React frontend
│   ├── components/            # React components
│   │   ├── ChatInterface.tsx  # Chat interface component
│   │   ├── FileUpload.tsx     # File upload component
│   │   └── ...
│   ├── App.tsx               # Main application component
│   └── ...
│
├── public/                    # Static files
└── README.md                 # Project documentation
```
---

## Model Choice and Rationale

### Model Used
- **ResNet18** (from the `timm` library)
  - **Input Channels:** 1 (grayscale MRI images)
  - **Number of Classes:** 3 (Healthy, ACL Injury, Meniscus Tear)
  - **Loss Function:** Focal Loss (α=1, γ=2)
  - **Optimizer:** AdamW with learning rate 1e-4
  - **Learning Rate Scheduler:** ReduceLROnPlateau

### Why ResNet18?
- **Proven Performance:** Widely used in medical imaging for strong generalization.
- **Residual Connections:** Prevent vanishing gradients, enabling deeper networks and better feature extraction.
- **Lightweight:** Efficient for real-time inference in web apps.
- **Transfer Learning:** Pretrained weights can be leveraged, but here the model is trained from scratch for single-channel input.

### Training Details
- **Epochs:** 25 (with early stopping)
- **Batch Size:** 16
- **Image Size:** 224x224 pixels
- **Data Augmentation:** Random rotation (±10°), horizontal flip
- **Class Weights:** Automatically balanced to handle class imbalance
- **Validation Split:** 20% of training data

### Data Preprocessing
- **Resize:** All images resized to (224, 224) pixels
- **Grayscale Conversion:** MRI images are single-channel
- **Normalization:** Mean = `[0.485]`, Std = `[0.229]` (chosen for MRI intensity distribution)
- **Data Augmentation:**
  - Random rotation (±10°)
  - Random horizontal flip

### Training Pipeline
1. **Data Loading:** Load and preprocess MRI images from `.pck` files
2. **Data Splitting:** 80% training, 20% validation
3. **Model Initialization:** ResNet18 with custom head
4. **Training Loop:**
   - Forward pass
   - Loss computation (Focal Loss)
   - Backpropagation
   - Learning rate scheduling
   - Model checkpointing
5. **Evaluation:** Compute metrics on validation set

---

## Model Performance

### Classification Report
```
              precision    recall  f1-score   support

           0       0.92      0.93      0.92      3349
           1       0.74      0.74      0.74       900
           2       0.84      0.70      0.76       277

    accuracy                           0.88      4526
   macro avg       0.83      0.79      0.81      4526
weighted avg       0.88      0.88      0.88      4526
```

### Key Metrics
- **Overall Accuracy:** 88%
- **Class 0 (Healthy) F1-Score:** 0.92
- **Class 1 (ACL Injury) F1-Score:** 0.74
- **Class 2 (Meniscus Tear) F1-Score:** 0.76
- **Dataset Size:** 4,526 samples

---

### Backend (FastAPI)
- **Model Serving:**
  - Loads trained ResNet18 model
  - Handles image preprocessing and inference
- **Endpoints:**
  - `POST /api/upload-mri`: Process MRI files and return predictions
  - `GET /health`: Health check endpoint
- **Error Handling:**
  - Invalid file formats
  - Model loading errors
  - Input validation

### Frontend (React + TypeScript)
- **Components:**
  - File upload with drag-and-drop support
  - Real-time chat interface
  - Analysis results visualization
- **State Management:** React hooks
- **Styling:** Tailwind CSS
- **Responsive Design:** Works on desktop and tablet

### End-to-End Flow
1. User uploads MRI images via the web interface
2. Frontend sends images to the backend API
3. Backend preprocesses images and runs inference using ResNet18
4. Predictions (class and confidence) are returned to the frontend
5. Frontend displays the results in a user-friendly format

### Deployment & Maintenance
- **Model Versioning:** Track model performance and versions
- **Monitoring:** Log API requests and model performance
- **Scaling:** Containerized deployment with Docker
- **CI/CD:** Automated testing and deployment pipeline

### Future Improvements
- **Model Upgrades:** Experiment with deeper ResNets or Vision Transformers
- **Explainability:** Add Grad-CAM visualizations
- **Multi-class/Multi-label Support:** Detect multiple injuries per image
- **Integration:** Connect with hospital PACS systems
- **Performance Optimization:** Quantization and model pruning

---

## How to Use KneeOps

### Prerequisites
- Node.js (v16 or later)
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/KneeOps.git
   cd KneeOps
   ```

2. **Set up the Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the Frontend**
   ```bash
   cd ..
   npm install
   ```

### Running the Application

#### Option 1: Using the start script (Recommended)
```bash
# Make the script executable if needed
chmod +x start.sh

# Run the application
./start.sh
```

#### Option 2: Manual Start

1. **Start the Backend**
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   ```
   The backend will start at `http://localhost:8000`

2. **Start the Frontend** (in a new terminal)
   ```bash
   cd frontend
   npm start
   ```
   The frontend will open automatically in your default browser at `http://localhost:3000`

### Using the Application

1. **Upload an MRI Scan**
   - Click the "Upload MRI" button or drag and drop a `.pck` file
   - The system will process the scan and display the analysis

2. **View Results**
   - The chat interface will show the analysis of your MRI scan
   - Results include the predicted condition and confidence level
   - For multiple scans, you can navigate between them using the interface

3. **Ask Questions**
   - Use the chat interface to ask questions about the results
   - The AI will provide additional information and explanations

---

### Troubleshooting

- **Backend not starting**: Ensure all dependencies are installed and port 8000 is available
- **Frontend not connecting to backend**: Check that the backend is running and the API URL is correctly set in the frontend configuration
- **Upload errors**: Ensure the file is in the correct `.pck` format

### Stopping the Application
- Press `Ctrl+C` in both terminal windows to stop the frontend and backend servers
- If using the start script, you can stop both services with:
  ```bash
  kill $BACKEND_PID $FRONTEND_PID
  ```

---

### References
- [Semi-automated detection of anterior cruciate ligament injury from MRI](https://doi.org/10.1016/j.cmpb.2016.12.006)
- [Kaggle Dataset](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset)
- [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.](https://arxiv.org/abs/1512.03385)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [ResNet and its application to medical image processing](https://pubmed.ncbi.nlm.nih.gov/37320940/)