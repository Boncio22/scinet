import torch
import pickle
import numpy as np
import torch.nn as nn


class SciNet(nn.Module):

    def __init__(self, input_size, latent_size, input2_size, output_size,
                 encoder_units=[100, 100], decoder_units=[100, 100]):
        super().__init__()
        encoder = Encoder(input_size, encoder_units, latent_size)
        decoder = Decoder(latent_size + input2_size, decoder_units,
                          output_size)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

    def forward(self, x, t, epsilon=0):
        mu, log_sigma = self.encoder(x)

        if self.training:
            epsilon = torch.randn(log_sigma.shape).cuda()
        encoded = mu + torch.exp(log_sigma) * epsilon
        decoded = self.decoder(x=encoded, t=t)

        return mu, log_sigma, decoded


class Encoder(nn.Module):

    def __init__(self, input_size, units, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=units[0]),
            nn.ELU(),
            nn.Linear(units[0], units[1]),
            nn.ELU(),
            nn.Linear(units[1], 2 * latent_size)
        )

    def forward(self, x):
        out = self.net(x)
        log_sigma = out[:, :self.latent_size]
        mu = out[:, self.latent_size:]

        return mu, log_sigma


class Decoder(nn.Module):

    def __init__(self, input_size, units, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=units[0]),
            nn.ELU(),
            nn.Linear(units[0], units[1]),
            nn.ELU(),
            nn.Linear(units[1], output_size)
        )

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=-1))


if __name__ == '__main__':
    scinet = SciNet(input_size=10, latent_size=2, input2_size=1, output_size=1)
    out = scinet(x=torch.ones((2, 10)), t=torch.ones((2, 1)))
    print(out)
