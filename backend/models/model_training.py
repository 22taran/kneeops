import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

# ----------------------------
# FOCAL LOSS DEFINITION
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

# ----------------------------
# Load and Preprocess Metadata
# ----------------------------
data_dir = '/kaggle/input/kneemridataset/'
metadata_path = os.path.join(data_dir, 'metadata.csv')
metadata_df = pd.read_csv(metadata_path)

label_encoder = LabelEncoder()
metadata_df['encoded_label'] = label_encoder.fit_transform(metadata_df['aclDiagnosis'])

filename_to_path = {
    file: os.path.join(root, file)
    for root, _, files in os.walk(data_dir)
    for file in files if file.endswith('.pck')
}

associated_data = []
for _, row in metadata_df.iterrows():
    path = filename_to_path.get(row['volumeFilename'])
    if path and os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
        for img in data:
            associated_data.append((img, row['encoded_label']))

# ----------------------------
# Dataset Definition
# ----------------------------
class MRIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if isinstance(img, np.ndarray):
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ----------------------------
# Transforms and Data Splits
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

train_data, test_data = train_test_split(
    associated_data, test_size=0.2, random_state=42,
    stratify=[item[1] for item in associated_data]
)

train_dataset = MRIDataset(train_data, transform=train_transform)
test_dataset = MRIDataset(test_data, transform=test_transform)

train_labels = [label for _, label in train_data]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor([class_weights[label] for label in train_labels], dtype=torch.float)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# ----------------------------
# Model Setup
# ----------------------------
num_classes = len(label_encoder.classes_)
model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes, in_chans=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ----------------------------
# Training Function
# ----------------------------
def train_with_improvements(model, train_loader, test_loader, device, criterion, optimizer, scheduler,
                            num_epochs=25, patience=5, model_save_path='best_knee_model.pth'):

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(" ✅ Best model saved.")
        else:
            patience_counter += 1
            print(f" ⏳ No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(" ⛔ Early stopping.")
                break

    return model, train_losses, val_losses, val_accuracies

# ----------------------------
# Run Training
# ----------------------------
model, train_losses, val_losses, val_accuracies = train_with_improvements(
    model, train_loader, test_loader, device,
    criterion, optimizer, scheduler,
    num_epochs=25, patience=5, model_save_path='best_knee_model.pth')

# ----------------------------
# Evaluate Model
# ----------------------------
model.load_state_dict(torch.load('best_knee_model.pth'))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n Classification Report:")
#print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
print(classification_report(all_labels, all_preds, target_names=[str(cls) for cls in label_encoder.classes_]))


print("\n Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ----------------------------
# Plot Training Curves
# ----------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
