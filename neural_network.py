from torch import nn as nn
from torch.nn import functional as F


class NeuralNetwork(nn.Module):
    """ A Neural Network with two hidden layers. """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output = F.log_softmax(self.layer3(x))  # use log or just softmax ?
        return output
