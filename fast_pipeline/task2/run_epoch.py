from enum import Enum

import time

import torch
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


def run_epoch(data_mode: DataMode, data_path: str, k: int = 100) -> list:
    if data_mode == DataMode.BRAIN:
        brain_dataset = BrainDataset(data_path)
        data_loader = DataLoader(brain_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    elif data_mode == DataMode.BIG_BRAIN:
        brain_dataset = BigBrainDataset(data_path)
        data_loader = DataLoader(brain_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 collate_fn=collate_fn, pin_memory=True)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        brain_dataset = UltraDuperBigBrainDataset(data_path, k=k)
        sampler = UltraDuperBigBrainBatchSampler(BATCH_SIZE, brain_dataset.bins,
                                                 brain_dataset.len_to_inds, len(brain_dataset))
        data_loader = DataLoader(brain_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                 collate_fn=collate_fn, batch_sampler=sampler)
    else:
        raise ValueError

    device = torch.device("cuda:0")

    model = MiniGPT2(brain_dataset.vocab_len).to(device)

    # warm-up
    model.train()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Warm-up")
    for i, (src, target) in pbar:
        if i > 10:
            break
        torch.cuda.synchronize()
        src = src.to(device)
        _ = model(src)

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    times = []
    for i, (src, target) in pbar:
        torch.cuda.synchronize()
        src = src.to(device)
        start = time.perf_counter()
        _ = model(src)
        end = time.perf_counter()
        times.append(end - start)

    return times
