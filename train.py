# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pipeline import xception_model, load_model # Assuming your pipeline.py has the model definition

# 1. Define Hyperparameters and Device
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define Data Transformations
# These must match the transformations used for inference in pipeline.py
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. Load the Dataset
train_dataset_path = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/train'
val_dataset_path = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/val'

train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 4. Load the Pre-trained Model
# The `load_model` function in pipeline.py already loads the ImageNet weights
# and replaces the final linear layer.
model = load_model(checkpoint_path=None, device=device)

# Freeze early layers to prevent them from being updated
# This is a common practice for fine-tuning to preserve learned features.
for name, param in model.named_parameters():
    # Unfreeze the last_linear layer and the last block for fine-tuning
    if "last_linear" in name or "conv4" in name or "block12" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 5. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# 6. Training and Validation Loop
print("Starting fine-tuning...")
best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # 7. Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # Save the best model checkpoint
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'deepfake_xception_best.pth')
        print("Saved new best model checkpoint.")

print("Fine-tuning complete.")