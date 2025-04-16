import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Define different activation functions
activations = {
    'erf': lambda: nn.ModuleDict({'act': nn.Identity(), 'func': lambda x: torch.erf(x)}),
    'sigmoid': lambda: nn.Sigmoid(),
    'tanh': lambda: nn.Tanh(),
    'relu': lambda: nn.ReLU()
}

# Neural network class that accepts an activation function
class Net(nn.Module):
    def __init__(self, activation_name):
        super(Net, self).__init__()
        self.activation_name = activation_name
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.set_activation(activation_name)

    def set_activation(self, name):
        if name == 'erf':
            self.act = lambda x: torch.erf(x)
        elif name == 'sigmoid':
            self.act = nn.Sigmoid()
        elif name == 'tanh':
            self.act = nn.Tanh()
        elif name == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

# Training loop
def train_model(activation_name, epochs=10):
    model = Net(activation_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        acc = correct / total
        losses.append(avg_loss)
        accuracies.append(acc)
        print(f"[{activation_name}] Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    return losses, accuracies

# Train all models and store results
results = {}
epochs = 10
for act_name in activations.keys():
    losses, accuracies = train_model(act_name, epochs=epochs)
    results[act_name] = {'loss': losses, 'acc': accuracies}

# Plotting
colors = {
    'erf': 'blue',
    'sigmoid': 'red',
    'tanh': 'green',
    'relu': 'gold'
}

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
for name, data in results.items():
    plt.plot(data['loss'], label=name, color=colors[name])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
for name, data in results.items():
    plt.plot(data['acc'], label=name, color=colors[name])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

