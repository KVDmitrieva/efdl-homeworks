from numpy import nonzero
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm

import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer,
                device: str, log_metrics: bool = False):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        if log_metrics:
            wandb.log(
                {
                    'train_loss': loss_ema,
                    'learning_rate': optimizer.defaults['lr']
                }
            )
        pbar.set_description(f"loss: {loss_ema:.4f}")


def generate_samples(model: DiffusionModel, device: str, path: str, x=None):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device, x=x)
        grid = make_grid(samples, nrow=4, normalize=True)   # normalize
        save_image(grid, path)

        if x is not None:
            input_image = to_pil_image(make_grid(x, nrow=4))
            wandb.log(
                {
                    "Input": wandb.Image(input_image),
                    "Generated images": wandb.Image(grid),
                }
            )
