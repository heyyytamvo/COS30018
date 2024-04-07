import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):

    """
     This constructor takes a configuration list config and a latent_dim parameter. It sets up the encoder and decoder networks using 
     linear layers and ReLU activations. The encoder network ends with two linear layers that output the mean (mu) and log variance (logVar) of the latent space distribution.
    """

    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )       
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        ) 

        self.decoder = nn.Sequential(*modules)

    """
    This function takes an input x and passes it through the encoder network to obtain the mean and log variance of the latent distribution.
    """

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    """
    This function takes a latent representation x and reconstructs the input data by passing it through the decoder network.
    """

    def decode(self, x):
        result = self.decoder(x)
        return result
    
    """
    This function takes the mean and log variance of the latent distribution and applies the reparameterization trick, 
    which allows the model to backpropagate through random sampling. 
    It's used during training to ensure that the latent space has good properties that allow for generating new data.
    """

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    """
    This is the function called during the forward pass of the network. 
    It encodes the input, reparameterizes the encoding, and then decodes it to produce the reconstructed output. 
    It returns the reconstructed output, the latent representation z, the mean mu, and the log variance logVar.
    """

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar
