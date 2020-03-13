import torch
from torch import nn
from torch.nn import functional as F


class Conv_NeuralNetwork(nn.Module):
    """ A Convolutional Neural Network with 2 cov layers and a fully connected one layers. """

    def __init__(self, conv1_in_chan, conv1_out_chan, conv2_out_chan, conv_kernel_size, pool_kernel_size,
                 hidden_size1, hidden_size2, output_size):
        super(Conv_NeuralNetwork, self).__init__()
        cov_padding = (conv_kernel_size - 1) // 2
        pool_padding = (pool_kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(conv1_in_chan, conv1_out_chan, conv_kernel_size, padding=cov_padding)
        self.conv2 = nn.Conv2d(conv1_out_chan, conv2_out_chan, conv_kernel_size, padding=cov_padding)
        self.maxpoool = nn.MaxPool2d(pool_kernel_size, stride=2, padding=pool_padding)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):

        x = self.maxpoool(F.relu(self.conv1(x)))
        x = self.maxpoool(F.relu(self.conv2(x)))
        # x = F.dropout2d(self.conv2(x))  # dropout p=0.5 by default
        # x = self.maxpoool(F.relu(x))
        # x = self.maxpoool(F.relu(F.dropout2d(self.conv2(x))))  # the above 2 lines in 1

        pixels_in_image = int((28 / 2 / 2)**2 * 20)  # 980
        x = x.view(-1, pixels_in_image)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        output = F.log_softmax(self.fc2(x))
        return output


class TutorialCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
