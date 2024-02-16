from typing import Optional

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
        with open(file_path, "w") as f:
            for i, line in f:
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
        super.__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        token = self.data[idx]
        return token[:-1], token[-1]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        super.__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        pass


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

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
