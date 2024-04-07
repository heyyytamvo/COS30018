import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    """
    The constructor sets up the network with convolutional layers (Conv1d) and linear layers (Linear). 
    It also includes batch normalization (BatchNorm1d) to stabilize learning, and activation functions (LeakyReLU, ReLU, Sigmoid).
    """

    def __init__(self, sig = True):
        super().__init__()
        self.sig = sig
        self.conv1 = nn.Conv1d(4, 32, kernel_size = 3, stride = 1, padding = 'same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 3, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 'same')

        """
        These layers process one-dimensional data (hence Conv1d). They are configured to maintain the same output length as the input length due to the padding='same' setting.
        """

        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)           # batch normalization layer to stabilize learning by normalizing the input to each layer
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)

        """
        These linear layers transform the data from the convolutional layers into a different space, eventually leading to a single output value.
        """

        self.leaky = nn.LeakyReLU(0.01)

        """
        LeakyReLU is used for non-linear processing with a small slope for negative values to avoid dead neurons. 
        ReLU is a common activation function that introduces non-linearity. Sigmoid squashes the output to a range between 0 and 1, which is useful for binary classification tasks.
        """

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    """
    This method defines the forward pass of the network. 
    It processes the input x through the convolutional and linear layers, applying the activation functions and batch normalization as specified. 
    The final output is either passed through a sigmoid function (if self.sig is True) to give a probability-like output, 
    or it is returned directly from the last linear layer (if self.sig is False).
    """

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        if self.sig:
            return self.sigmoid(self.linear3(out_2))
        else:
            return self.linear3(out_2)