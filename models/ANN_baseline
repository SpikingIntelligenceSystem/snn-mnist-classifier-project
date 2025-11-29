import torch.nn as nn
# Imports

class MNIST_ANN(nn.Module):

    def __init__(self, beta=0.9, spike_grad=None):
        super().__init__()
        neuron_inputs = 784  # 28*28
        neuron_hidden_1 = 256
        neuron_hidden_2 = 128
        neuron_outputs = 10
        # Network neuron counts^
        self.layer_1 = nn.Linear(neuron_inputs, neuron_hidden_1)
        self.layer_2 = nn.Linear(neuron_hidden_1, neuron_hidden_2)
        self.layer_3 = nn.Linear(neuron_hidden_2, neuron_outputs)
        # Connections defined^
        self.act = nn.ReLU()
        # Non spiking network defined^

    def forward(self, x):
        batch = x.size(0)
        flat_x = x.view(batch, -1)
        # Flatten input image

        cur1 = self.layer_1(flat_x)
        spk1 = self.act(cur1)
        cur2 = self.layer_2(spk1)
        spk2 = self.act(cur2)
        logits = self.layer_3(spk2)
        # Processes data through network
        return logits
