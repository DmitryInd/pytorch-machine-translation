import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, max_len):
        """
        emb_size - размер эмбеддингов (E - embedding)
        max_len - длинна контекста (S - sequence)
        """
        super(PositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, emb_size)  # (S, E)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (S, 1)
        # 1/1000^(2i/emb_size): (E)
        denominator = torch.exp(-np.log(10000) * torch.arange(0, emb_size, 2).float() / emb_size)
        # PE[pos, 2i] = sin(pos/1000^(2i/emb_size)): (S, E)
        pos_encoding[:, 0::2] = torch.sin(positions * denominator)
        # PE[pos, 2i] = cos(pos/1000^(2i/emb_size)): (S, E)
        pos_encoding[:, 1::2] = torch.cos(positions * denominator)
        # (S, B, E), B - batch
        pos_encoding = pos_encoding.unsqueeze(1)
        # Saving parameters to buffer without gradients
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor):
        """
        token_embedding - тензор матрицы эмбеддингов: (S, B, E)
        """
        return token_embedding + self.pos_encoding[:token_embedding.size(0)]
