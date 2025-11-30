import torch
import torch.nn as nn
from snntorch import spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from models.MNIST_single_step_SNN_baseline import MNIST_1STEP_SNN
from pathlib import Path
import json
# Establish imports

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
# Establish data results path

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# For higher levels of consistency

"""
Train + evaluate the single step SNN baseline.

Run from repo root:
    python -m scripts.train_single_step_baseline
"""

batch_size = 128
num_epochs = 10
learn_rate = 1e-3
# Set training parameters

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# Attempts to use CUDA if available, if unavailable uses CPU


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Transforms input into readable data for network

training_data = datasets.MNIST(
    root="/tmp/data/mnist", train=True, download=True, transform=transform)
testing_data = datasets.MNIST(
    root="/tmp/data/mnist", train=False, download=True, transform=transform)
# Dataset downloads
training_loader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(
    testing_data, batch_size=batch_size, shuffle=False)
# Dataset loaders

net = MNIST_1STEP_SNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    net.parameters(), learn_rate)
# Define network model, loss, and optimizer model


def train_SNN(model, training_loader, device, num_epochs):
    epoch_loss = []
    for epoch in range(num_epochs):
        iteration = 0
        model.train()
        loss_total = []
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)  # Forward pass
            loss_batch = criterion(logits, labels)  # Loss computation
            loss_batch.backward()  # Backpropagation
            optimizer.step()  # Optimizer updates weights
            iteration += 1  # Increases iteration count by 1 after iteration
            loss_total.append(loss_batch.item())
            if iteration % 50 == 0:
                mean_loss = sum(loss_total) / len(loss_total)
                print(f"Epoch: {epoch + 1}, Iteration: {iteration}, "
                      f"Batch Loss: {loss_batch.item():.4f}, "
                      f"Mean Loss: {mean_loss:.4f}")
                # Prints epoch, loss, and iterations after every 50 iterations
        epoch_mean = sum(loss_total) / len(loss_total)
        epoch_loss.append(epoch_mean)
    return epoch_loss   # Returns epoch loss values for plotting
# Model training script


def acc_eval_SNN(model, testing_loader, device):
    model.eval()
    correct_img = 0
    total_img = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct_img += (preds == labels).sum().item()
            total_img += labels.size(0)
        analysis_accuracy = correct_img / total_img
        print(f"Total Test Accuracy: {(analysis_accuracy * 100):.3f}%")
        return analysis_accuracy
# Model overall evaluation script


# Engage testing
if __name__ == "__main__":
    epoch_loss = train_SNN(net, training_loader, device, num_epochs)
    loss_path = RESULTS_DIR / "1step_snn_loss.json"
    with loss_path.open("w") as f:
        json.dump(epoch_loss, f)
        # For plotting loss
    acc_eval_SNN(net, testing_loader, device)
    torch.save(net.state_dict(), "mnist_1Step_SNN_baseline.pt")
    print("Single Step SNN Baseline Saved")
    # Save trained Single Step SNN baseline
