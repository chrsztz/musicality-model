import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

# Define the dataset path (replace with your actual dataset path)
dataset_path = 'augmented_waste_classification'  # Example: 'C:/Users/YourName/waste_images'

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load the dataset from folders
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class WasteClassifierCNN(nn.Module):
    def __init__(self):
        super(WasteClassifierCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channels (RGB), 16 output channels
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 input channels, 32 output channels
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 512)  # After two pooling layers: 224 / 4 = 56
        self.fc2 = nn.Linear(512, 10)  # 10 output classes (one for each label)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected -> ReLU
        x = self.fc2(x)  # Output layer
        return x

# Check for GPU availability
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = WasteClassifierCNN().to(device)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print progress
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'waste_classifier.pth')
print("Model saved as 'waste_classifier.pth'")

# Example: How to load and use the model for inference
"""
# Load the model
model = WasteClassifierCNN().to(device)
model.load_state_dict(torch.load('waste_classifier.pth'))
model.eval()

# Preprocess a new image (example)
new_image = transform(your_image).unsqueeze(0).to(device)  # Add batch dimension
with torch.no_grad():
    output = model(new_image)
    _, predicted = torch.max(output, 1)
    predicted_label = dataset.classes[predicted.item()]
    print(f'Predicted class: {predicted_label}')
"""