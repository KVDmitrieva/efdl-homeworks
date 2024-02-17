# code sources:
# - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# - PyTorch source code
import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer


class MiniGPT2(nn.Module):
    def __init__(self, ntoken: int, d_model: int = 512, nhead: int = 8, d_hid: int = 1024,
                 dropout: float = 0.5, pad_idx: int = 0):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        # в нашем контексте декодер=энкодер
        self.decoder = TransformerEncoderLayer(d_model, nhead, d_hid, batch_first=True)
        self.pad_idx = pad_idx

    def forward(self, src: Tensor) -> Tensor:
        pad_mask = (src == self.pad_idx).to(src.device)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = generate_square_subsequent_mask(src.shape[1]).to(src.device)

        output = self.decoder(src, src_mask=src_mask, src_key_padding_mask=pad_mask)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
