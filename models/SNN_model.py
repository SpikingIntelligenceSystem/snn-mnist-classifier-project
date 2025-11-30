import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
# Imports

class MNIST_SNN(nn.Module):
    def __init__(self, beta=0.9, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.atan()
        neuron_inputs = 784  # 28*28
        neuron_hidden_1 = 256
        neuron_hidden_2 = 128
        neuron_outputs = 10
        # Network neuron counts^
        self.layer_1 = nn.Linear(neuron_inputs, neuron_hidden_1)
        self.layer_2 = nn.Linear(neuron_hidden_1, neuron_hidden_2)
        self.layer_3 = nn.Linear(neuron_hidden_2, neuron_outputs)
        # Connections defined^
        self.network_1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.network_2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.network_3 = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        # Output= allows for the return of spikes and membrane.
        # Network layers defined^

    def forward(self, spk_in):
        spk_rec = []
        mem_rec = []
        num_s, _, _ = spk_in.shape
        utils.reset(self)  # Reset hidden state

        for s in range(num_s):
            initial = spk_in[s]
            cur1 = self.layer_1(initial)
            spk1 = self.network_1(cur1)
            cur2 = self.layer_2(spk1)
            spk2 = self.network_2(cur2)
            cur3 = self.layer_3(spk2)
            spk3, mem3 = self.network_3(cur3)
            spk_rec.append(spk3)
            mem_rec.append(mem3)
            # Processes data through network in a certain time period (num_s)

        spk_rec = torch.stack(spk_rec)
        # Spikes over time
        mem_rec = torch.stack(mem_rec)
        # Membrane over time
        return spk_rec, mem_rec
