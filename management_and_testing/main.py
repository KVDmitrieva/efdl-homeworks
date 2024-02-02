import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import argparse
import hydra
import omegaconf
import wandb

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(config_name="config")
def main(cfg, wandb_key: str, device: str, num_epochs: int = 100):
    wandb.login(key=wandb_key)
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ddpm = DiffusionModel(
        eps_model=UnetModel(cfg.model.unet.in_channels, cfg.model.unet.out_channels, hidden_size=cfg.model.unet.hidden_size),
        betas=(cfg.model.beta1, cfg.model.beta2),
        num_timesteps=cfg.model.num_timestamps,
    )
    ddpm = ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.training.lr)

    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-k",
        "--key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    args = args.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(wandb_key=args.key, device=device)
