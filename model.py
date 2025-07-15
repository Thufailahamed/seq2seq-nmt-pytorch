import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6

    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000

    max_src_len: int = 512
    max_tgt_len: int = 512
    learn_pos_embed: bool = False

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, requires_grad=False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

    def _build_positional_encodings(self):

        encoding = torch.zeros(self.max_len, self.embed_dim, dtype=torch.float)
        postion_idx = torch.arange(0, self.max_len, dtype=torch.float).reshape(-1,1)
        embed_dim_skip_idx = torch.arange(0, self.embed_dim, step=2, dtype=torch.float)
        
        encoding[:, 0::2] = torch.sin(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))
        encoding[:, 1::2] = torch.cos(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))

        encoding = nn.Parameter(encoding, requires_grad=self.requires_grad)

        return encoding

    def forward(self, x):
        seq_len = x.shape[1]

        encodings = self.encodings[:seq_len]

        x = x + encodings

        return x