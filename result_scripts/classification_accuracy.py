import matplotlib.pyplot as plt

# Replace accuracies with your final measured accuracies
model_names = ["ANN", "1-Step SNN", "SNN (T=100)"]
accuracies = [0.98, 0.975, 0.972]  # as fractions (0–1)

plt.figure()
plt.bar(model_names, [a * 100 for a in accuracies])
plt.ylabel("Test Accuracy (%)")
plt.ylim(90, 100)
plt.title("MNIST Classification Accuracy")

plt.tight_layout()
plt.savefig("results/classification_accuracy.png", dpi=300)
# plt.show()  # uncomment if running interactively
