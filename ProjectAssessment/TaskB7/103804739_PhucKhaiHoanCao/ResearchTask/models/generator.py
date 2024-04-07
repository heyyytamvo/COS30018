import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    """
    The constructor initializes the Generator class as a subclass of nn.Module. 
    It sets up the device to use CUDA if available and requested (use_cuda is True), otherwise it uses the CPU. 
    It then defines three GRU (Gated Recurrent Unit) layers and three linear layers, along with a dropout layer for regularization and a sigmoid activation function for the output.
    """
    def __init__(self, input_size, use_cuda):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)       # process sequences of data
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)                             # standard fully connected neural network layers to transform the input into the desired output size.
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)                  # dropout layer to prevent overfitting that randomly sets a fraction of the input units to 0 at each update during training time
        self.sigmoid = nn.Sigmoid()                     # sigmoid activation function to squash the output to a range between 0 and 1

    """
    This method defines the forward pass of the neural network. 
    It takes an input x and sequentially passes it through the GRU layers, applying dropout after each. 
    The final output of the GRUs is then passed through the linear layers, and the sigmoid function is applied to the final output.
    """

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 1024).to(self.device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(self.device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(self.device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out_6 = self.linear_3(out_5)
        out = self.sigmoid(out_6)
        return out