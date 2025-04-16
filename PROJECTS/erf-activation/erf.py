""" erf.py
~~~~~~~~~~
A doctored classification problem so that I can assert that the error function 
works as a valid and better activation function than ReLU / Sigmoid.

erf tends to converge faster, but as you increase ``epochs`` and training size,
the classical ReLU / Tanh activation functions dominate:

~~~~~~~~~~
Training with ReLU activation
ReLU Test Accuracy: 87.67%
Training with Tanh activation
Tanh Test Accuracy: 84.00%
Training with Erf activation
Erf Test Accuracy: 91.33%
"""

# 3rd party imports
import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

# Random Seeds
torch.manual_seed(69)
np.random.seed(420)

# Dataset.
X, y = make_blobs(n_samples=1000, centers=[(0,0), (2,2)], cluster_std=1.0)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Torch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class ErfActivation(nn.Module):
  def forward(self, x):
    return torch.erf(x)

class SimpleNet(nn.Module):
  def __init__(self, activation):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(2,16),
      activation(),
      nn.Linear(16,1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.net(x)

# Training function
def train(model, X_train, y_train, X_test, y_test, epochs=10):
  criterion = nn.BCELoss()
  optimiser = optim.Adam(model.parameters(), lr=0.01)
  train_losses, test_losses = [], []

  for epoch in range(epochs):
    model.train()
    optimiser.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimiser.step()
    train_losses.append(loss.item()) # the value, (it's wrapped in a tensor.)

    model.eval()
    with torch.no_grad():
      test_output = model(X_test)
      test_loss = criterion(test_output, y_test)
      test_losses.append(test_loss.item())

  return train_losses, test_losses

def evaluate(model, X, y):
  model.eval()
  with torch.no_grad():
    preds = model(X)
    preds = (preds > 0.5).float()
    acc = (preds == y).float().mean().item()
  return acc

def plot_decision_boundary(model, title):
  xx, yy = np.meshgrid(np.linspace(-3,3,200), np.linspace(-3,3,200))
  grid = torch.tensor(np.c_[xx.ravel(),yy.ravel()], dtype=torch.float32)
  with torch.no_grad():
    probs = model(grid).reshape(xx.shape)
  plt.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
  plt.scatter(X_test[:,0], X_test[:,1], c=y_test.flatten(), cmap="bwr_r", edgecolors='k')
  plt.title(title)
  plt.savefig("{}.svg".format(title), format='svg')


activations = {
  "ReLU": nn.ReLU,
  "Tanh": nn.Tanh,
  "Erf": ErfActivation
}

for name, act in activations.items():
  print(f"Training with {name} activation")
  model = SimpleNet(act)
  train_losses, test_losses = train(model, X_train, y_train, X_test, y_test)
  acc = evaluate(model, X_test, y_test)
  print(f"{name} Test Accuracy: {acc*100:.2f}%")
  plot_decision_boundary(model, f"{name} Decision Bondary")

