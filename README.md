# snn-mnist-classifier-project

This repository contains a personal project exploring **spiking neural networks (SNNs)** using [PyTorch](https://pytorch.org/) and [snnTorch](https://snntorch.readthedocs.io/).  
The goal is to build and train a simple SNN to classify handwritten digits from the **MNIST** dataset and compare its behavior to a standard feedforward neural network (ANN) baseline.

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

## Status

**Work in progress.** 

---

## Repository Structure

Planned structure for this project (subject to change):

```text
snn-mnist-classifier-project/
├── models/
│   ├── snn_model.py          # SNN architecture definitions
│   └── ann_baseline.py       # ANN model for comparison
├── notebooks/
│   └── experimentation.ipynb     # experiments and visualizations
├── train_snn.py              # main SNN training script
├── train_ann_baseline.py     # ANN baseline training script
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
Training scripts are currently under development.

---

## Results
```markdown
No results yet - will be listed here when complete.

Planned metrics to report:

- SNN accuracy on the MNIST test set
- Accuracy of an ANN baseline for comparison
- Training/validation loss and accuracy curves
```

---

## License

This project is licensed under the terms of the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.
