import click
import torch
import numpy as np
import pylab as plt
from tqdm import tqdm
from model import SciNet
from data_loader import get_data


@click.command()
@click.option('-d', '--dataset', default='oscillator')
@click.option('-lr', '--learning_rate', default=1e-3)
@click.option('-e', '--epochs', default=100)
@click.option('--device', default='cuda')
@click.option('--model', type=click.Path(), default=None)
@click.option('--beta', default=1e-3)
def train(dataset: str,
          learning_rate: float,
          epochs: int,
          device: str,
          model: str,
          beta: float):
    def kl_divergence(means, log_sigma, target_sigma=1.):
        kl_div = target_sigma**-2 * means**2 + 2 * np.log(target_sigma)
        kl_div += -2 * log_sigma + torch.exp(2 * log_sigma) / target_sigma**2
        return torch.mean(torch.sum(kl_div, dim=1))

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = get_data(dataset)

    net_setup = {
        'oscillator': {
            'input_size': 50,
            'input2_size': 1,
            'latent_size': 2,
            'output_size': 1,
            'encoder_units': [100, 100],
            'decoder_units': [100, 100]
        },
        'collision': {
            'input_size': 30,
            'input2_size': 16,
            'latent_size': 1,
            'output_size': 2,
            'encoder_units': [150, 100],
            'decoder_units': [100, 150]
        }
    }[dataset]

    network = SciNet(**net_setup)
    network = network.to(device)

    if model:
        network.load_state_dict(torch.load(model))
        print("Restored weights")

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        for i, (x, t, y) in enumerate(train_loader):
            x, t, y = x.to(device), t.to(device), y.to(device)

            optimizer.zero_grad()

            mu, log_sigma, output = network(x, t)

            loss = mse_loss(output, y) + beta * kl_divergence(mu, log_sigma)
            loss.backward()

            torch.nn.utils.clip_grad.clip_grad_value_(network.parameters(),
                                                      10.0)

            optimizer.step()

        print(loss.item())

    torch.save(network.state_dict(), f'{dataset}.pth')


if __name__ == '__main__':
    train()
