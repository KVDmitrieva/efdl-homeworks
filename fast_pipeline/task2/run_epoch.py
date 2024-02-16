from enum import Enum

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from transformer import MiniGPT2
from dataset import *


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


BATCH_SIZE = 128
LR = 1e-4


def run_epoch(data_mode: DataMode, data_path: str) -> list:
    if data_mode == DataMode.BRAIN:
        brain_dataset = BrainDataset(data_path)
        data_loader = DataLoader(brain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    elif data_mode == DataMode.BIG_BRAIN:
        brain_dataset = BigBrainDataset(data_path)
        data_loader = DataLoader(brain_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        brain_dataset = BigBrainDataset(data_path)
        data_loader = DataLoader(brain_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 collate_fn=collate_fn, batch_sampler=UltraDuperBigBrainBatchSampler)
    else:
        raise ValueError

    device = torch.device("cuda:0")

    model = MiniGPT2(brain_dataset.vocab_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
    static_input = torch.randn(BATCH_SIZE, 640, device='cuda')
    static_target = torch.randn(BATCH_SIZE, device='cuda')

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = criterion(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    model.train()
    times = []
    for i, (src, labels) in pbar:
        torch.cuda.synchronize()
        src = src.to(device)
        start = time.perf_counter()
        _ = model(src)
        end = time.perf_counter()
        times.append(end - start)

    return times
