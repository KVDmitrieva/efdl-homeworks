import torch
from torch import nn
from tqdm.auto import tqdm

from task1.unet import Unet

from task1.dataset import get_train_data

from task1.scaler import CustomScaler


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scaler: CustomScaler | torch.cuda.amp.GradScaler,
    device: torch.device,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaler="custom", mode="static"):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # GradScaler just for test
    scaler = CustomScaler(mode) if scaler == "custom" else torch.cuda.amp.GradScaler()

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, device=device)
