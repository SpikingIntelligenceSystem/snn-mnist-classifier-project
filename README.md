# snn-mnist-classifier-project

This repository contains a personal project exploring **spiking neural networks (SNNs)** using [PyTorch](https://pytorch.org/) and [snnTorch](https://snntorch.readthedocs.io/).  
The goal is to build and train a simple SNN to classify handwritten digits from the **MNIST** dataset and compare its behavior to a standard artificial neural network (ANN) baseline.
I will also include a single step SNN as another baseline comparison that has relatively similar accuracy to the ANN baseline.

This project is part of my self-study in **neuromorphic computing, spiking neural networks, and machine learning**.

---

## Project Goals

- Implement a basic spiking neural network for MNIST digit classification.
- Experiment with:
  - Different spike encoding schemes (e.g., rate coding).
  - Simple neuron models (e.g., Leaky Integrate-and-Fire).
- Compare:
  - SNN performance and training dynamics vs. a standard ANN baseline.
- Build a clean, well-documented project that can be shared on my resume and GitHub.

---

## Technologies Used

- **Language:** Python
- **Frameworks:** PyTorch, snnTorch
- **Dataset:** MNIST (via `torchvision.datasets.MNIST`)
- **Environment:** VS Code / Python virtual environment

---

## Model Architectures

This project compares three related models on the MNIST handwritten digit dataset:
a standard feedforward ANN, a single–step SNN, and a temporal (multi–step) SNN.

### 1. Feedforward ANN Baseline

**Type:** Standard fully connected neural network  
**Input:** 28×28 grayscale image → flattened to 784-dim vector  

**Layer stack**

- Linear(784 → 256)
- ReLU
- Linear(256 → 128)
- ReLU
- Linear(128 → 10) → logits (passed to `CrossEntropyLoss`)

**Key details**

- No time dimension — each image is processed in a single forward pass.
- Optimizer: Adam (lr = 1e-3), batch size = 128, epochs = 10.
- Serves as the non-spiking baseline for accuracy and training behavior.

---

### 2. Single-Step SNN Baseline

**Type:** LIF-based SNN with a single simulation step  
**Input:** 28×28 image → flattened to 784-dim vector (no spike encoding)

**Layer stack**

- Linear(784 → 256) → LeakyLIF
- Linear(256 → 128) → LeakyLIF
- Linear(128 → 10) → LeakyLIF (with `output=True` to get membrane + spikes)

**Key details**

- Only **one** time step is simulated: the image is passed through the LIF layers once.
- The final layer’s **membrane potential** is used as the logits for `CrossEntropyLoss`.
- Same training setup as the ANN (Adam, batch size 128, epochs 10), but with spiking
  neurons replacing ReLU.

---

### 3. Temporal SNN (Multi-Step SNN)

**Type:** LIF-based SNN with rate-encoded inputs and multiple time steps  
**Input pipeline**

1. Image: 28×28 → flatten to 784
2. Rate encoding with `spikegen.rate` to produce a spike train:
   - Shape: **[T, batch, 784]**, where `T = num_steps` (e.g., 100)

**Layer stack per time step**

For each time step `t`:

- Linear(784 → 256) → LeakyLIF
- Linear(256 → 128) → LeakyLIF
- Linear(128 → 10) → LeakyLIF (with `output=True`)

**Readout**

- The network records membrane potentials `mem_rec` over all time steps:
  - Shape: **[T, batch, 10]**
- Classification logits are obtained by time-averaging:
  - `logits = mem_rec.mean(dim=0)` → shape **[batch, 10]**
- These logits are passed to `CrossEntropyLoss`.

**Training setup**

- Time steps: num_steps = 100
- Optimizer: Adam (lr = 1e-4)
- Batch size: 128
- Epochs: 5-10 (10 used for the accuracy comparison results)

---

**Data Flowchart For Temporal SNN**
```mermaid
flowchart LR
  img["MNIST image 28x28"]
  flat["Flatten to 784"]
  encode["Rate encoding: spikegen.rate(x, T)"]
  spikes["Spike train, shape: T x batch x 784"]
  lif1["Linear(784->256) + LIF"]
  lif2["Linear(256->128) + LIF"]
  lif3["Linear(128->10) + LIF"]
  avg["Average over time: mem_rec.mean(dim=0)"]
  logits["Class logits, shape: batch x 10"]

  img --> flat --> encode --> spikes
  spikes --> lif1 --> lif2 --> lif3 --> avg --> logits
```

---

## Status

**Comparing models.** 

---

## Repository Structure

Planned structure for this project (subject to change):

```text
snn-mnist-classifier-project/
├── models/
│   ├── snn_model.py          # SNN architecture definitions
│   ├── mnist_single_step_snn_baseline.py    # single step SNN model for comparison
│   └── mnist_ann_baseline.py       # ANN model for comparison
├── notebooks/
│   └── experimentation.ipynb     # experiments and visualizations
├── scripts/
│   ├── train_snn.py              # main SNN training script
│   ├── train_single_step_baseline.py    # single step SNN baseline training script
│   └── train_ann_baseline.py     # ANN baseline training script
├── requirements.txt          # Python dependencies
└── README.md                 # project documentation
```

---

## Setup

A basic Python environment is recommended, but not needed (e.g., virtualenv or venv).

Once `requirements.txt` is complete, dependencies can be installed with:

```bash
pip install -r requirements.txt
```

---

## Project Results

### Model Accuracy Results

| Model         | Timesteps | Epochs    | Test Accuracy | Model Type          |
|---------------|-----------|-----------|---------------|---------------------|
| ANN           | 1         | 10        | 98.0%         | ReLU ANN            |
| 1-step SNN    | 1         | 10        | 97–98%        | LIF SNN             |
| SNN Model     | 100       | 10        | 97%           | LIF SNN             |

---

## License

This project is licensed under the terms of the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.
