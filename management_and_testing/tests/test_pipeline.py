import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm = ddpm.to(device)      # fix device

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
@pytest.mark.parametrize(["hidden_size"], [[32], [64]])
def test_training(device, hidden_size, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=hidden_size),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm = ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    train_set = torch.utils.data.Subset(train_dataset, torch.arange(20))
    dataloader = DataLoader(train_set, batch_size=4, shuffle=True)

    if not os.path.exists('samples'):
        os.mkdir('samples')

    for i in range(3):
        train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")
