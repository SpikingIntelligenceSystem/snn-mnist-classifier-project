import torch
import torch.nn as nn
from snntorch import spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.SNN_model import MNIST_SNN
# Imports

"""
Train + evaluate the full SNN.

Run from repo root:
    python -m scripts.train_SNN
"""

batch_size = 128
num_steps = 100
num_epochs = 10
learn_rate = 1e-4
# Set training parameters

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# Attempts to use CUDA if available, if unavailable uses CPU


def spike_encoding(images, num_steps, device):
    images = images.to(device)
    # Moves images to device
    batch = images.size(0)
    flatten = images.view(batch, -1)
    # Flattens image
    spike = spikegen.rate(flatten, num_steps)
    # Encode to spikes
    return spike


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

net = MNIST_SNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    net.parameters(), learn_rate)
# Define network model, loss, and optimizer model


def train_SNN(model, training_loader, device, num_epochs, num_steps):
    for epoch in range(num_epochs):
        iteration = 0
        model.train()
        loss_total = []
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            spikes = spike_encoding(images, num_steps, device)
            # Encode images as spikes
            spk_rec, mem_rec = model(spikes)
            # Pass spikes through model
            logits = mem_rec.mean(dim=0)
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
# Model training script


def acc_eval_SNN(model, testing_loader, device, num_steps):
    model.eval()
    correct_img = 0
    total_img = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes = spike_encoding(images, num_steps, device)
            _, mem_rec = model(spikes)
            logits = mem_rec.mean(dim=0)
            preds = logits.argmax(dim=1)
            correct_img += (preds == labels).sum().item()
            total_img += labels.size(0)
        analysis_accuracy = correct_img / total_img
        print(f"Total Test Accuracy: {(analysis_accuracy * 100):.3f}%")
        return analysis_accuracy
# Model overall evaluation script


if __name__ == "__main__":
    train_SNN(net, training_loader, device, num_epochs, num_steps)
    acc_eval_SNN(net, testing_loader, device, num_steps)
    # Engage testing
    torch.save(net.state_dict(), "SNN_model.pt")
    print("SNN Model Saved")
    # Save trained SNN
