import json
from pathlib import Path
import matplotlib.pyplot as plt

"""
Running the plotting program.

Run from repo root:
    python -m result_scripts.plot_loss_curves
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_losses(file):
    path = RESULTS_DIR / file
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    ANN = load_losses("ANN_loss.json")
    single_step = load_losses("1step_snn_loss.json")
    SNN = load_losses("SNN_loss.json")
    # Retrieve loss data from JSON files
    if not any([ANN, single_step, SNN]):
        print("No loss curves found in results. Nothing to plot.")
        return

    plt.figure(figsize=(8, 5))

    if ANN is not None:
        epochs = range(1, len(ANN)+1)
        plt.plot(epochs, ANN, marker="o", label="ANN Loss")

    if single_step is not None:
        epochs = range(1, len(single_step)+1)
        plt.plot(epochs, single_step, marker="o", label="Single Step SNN Loss")

    if SNN is not None:
        epochs = range(1, len(SNN)+1)
        plt.plot(epochs, SNN, marker="o", label="SNN Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Vs. Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "training_loss_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    # plt.show() # uncomment if running interactively
    print(f"Saved loss curves to {out_path}.")


if __name__ == "__main__":
    main()
