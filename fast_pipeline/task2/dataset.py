from collections import defaultdict
from random import sample, choices

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

MAX_LENGTH = 640

__all__ = ["BrainDataset", "BigBrainDataset", "UltraDuperBigBrainDataset",
           "collate_fn", "UltraDuperBigBrainBatchSampler"]


class BaseBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(self.text_iterator(data_path), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        self.vocab_len = len(vocab)

        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in self.text_iterator(data_path)]
        self.data = [item[:max_length] for item in data if item.numel() > 1]

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    @staticmethod
    def text_iterator(file_path: str, limit=1e5):
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                yield line


class BrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        raw_token = self.data[idx]
        token = torch.zeros(self.max_length, dtype=torch.long)
        token[:len(raw_token) - 1] = raw_token[:-1]
        return token, raw_token[-1]


class BigBrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        token = self.data[idx]
        return token[:-1], token[-1]


class UltraDuperBigBrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, k: int = 640):
        super().__init__(data_path, max_length)

        self.len_to_inds = defaultdict(list)
        self.bins = {}

        for i, tokens in enumerate(self.data):
            n = len(tokens)
            self.len_to_inds[n].append(i)

        # длин максимум 640 <<< len(data)
        candidates = sorted(self.len_to_inds.keys())
        left, right = 0, 1
        bin_num = 0
        while right < len(candidates):
            if candidates[right] - candidates[left] <= k:
                right += 1
            else:
                self.bins[bin_num] = candidates[left:right]
                bin_num += 1
                left += 1
        self.bins[bin_num] = candidates[left:right]

    def __getitem__(self, idx: int):
        token = self.data[idx]
        return token[:-1], token[-1]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :return: tuple of padded sequences and corresponding training targets
    """
    max_len = max([len(tokens) for tokens, _ in batch])
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = []
    for i, (token, target) in enumerate(batch):
        tokens[i, :len(token)] = token
        targets.append(target)

    targets = torch.tensor(targets)
    return tokens, targets


class UltraDuperBigBrainBatchSampler(Sampler):
    def __init__(self, batch_size: int, bins: dict, inds: dict, len_data):
        self.bs = batch_size
        self.len = (len_data + batch_size - 1) // batch_size
        self.bin = bins
        self.inds = inds

    def __len__(self):
        return self.len

    def __iter__(self):
        bin_candidates = list(self.bin.keys())
        bin_candidates = choices(bin_candidates, k=self.len)
        for candidate in bin_candidates:
            inds = []
            for length in self.bin[candidate]:
                inds.extend(self.inds[length])
            yield sample(inds, min(self.bs, len(inds)))
