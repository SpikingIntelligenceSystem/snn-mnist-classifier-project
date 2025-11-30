import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
from models.SNN_model import MNIST_SNN

"""
Running the plotting program.

Run from repo root:
    python -m result_scripts.SNN_spike_raster
"""

num_steps = 100     # Enter same value as used in SNN training
batch_size = 1      # Only 1 image needed for plotting
RESULTS_DIR = Path("results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="/tmp/data/mnist",
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


def spike_encoding(images, num_steps, device):
    images = images.to(device)
    batch = images.size(0)
    flat = images.view(batch, -1)
    spikes = spikegen.rate(flat, num_steps)
    return spikes


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    net = MNIST_SNN().to(device)
    state_dict = torch.load("SNN_model.pt", map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    images, labels = next(iter(test_loader))
    labels = labels.item()

    spikes = spike_encoding(images, num_steps, device)

    with torch.no_grad():
        spk_rec, mem_rec = net(spikes)
    spk_single = spk_rec[:, 0, :]

    t_idx, neuron_idx = torch.nonzero(spk_single, as_tuple=True)

    plt.figure(figsize=(8, 4))
    plt.scatter(
        t_idx.cpu(),
        neuron_idx.cpu(),
        s=80,
        marker="|",
        linewidths=2
    )

    plt.xlabel("Time Step")
    plt.ylabel("Output Neuron")
    plt.yticks(range(10))
    plt.title(f"SNN Output Spike Raster")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = RESULTS_DIR / "snn_output_spikes.png"
    plt.savefig(out_path, dpi=300)
    plt.show()  # Uncomment if running interactively
    print(f"Save spike raster to: {out_path}")


if __name__ == "__main__":
    main()
