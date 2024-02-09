import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import hydra
import omegaconf
import wandb
from hydra.utils import instantiate

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(config_path='.', config_name="config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    wandb.config.update(omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    ddpm = DiffusionModel(
        eps_model=UnetModel(cfg.model.unet.in_channels, cfg.model.unet.out_channels,
                            hidden_size=cfg.model.unet.hidden_size),
        betas=(cfg.model.beta1, cfg.model.beta2),
        num_timesteps=cfg.model.num_timestamps,
    )
    ddpm = ddpm.to(device)

    train_transforms = instantiate(cfg.augmentations)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
                            shuffle=True)
    # optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.training.lr)
    optim = instantiate(cfg.optimizer, params=ddpm.parameters())

    fixed_input = torch.randn(8, *(cfg.model.unet.in_channels, 32, 32))
    if not os.path.exists('samples'):
        os.mkdir('samples')
    for i in range(cfg.training.num_epochs):
        train_epoch(ddpm, dataloader, optim, device, log_metrics=True)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")
        generate_samples(ddpm, device, f"samples/fixed_{i:02d}.png", fixed_input)


if __name__ == "__main__":
    main()
