import math

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from .cnn import Cnn

class TimestepEmbedder(nn.Module):
    r"""Time step embedder.
    
    References:
    [1] https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/unet/nn.py
    [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """
    def __init__(self, out_channels: int, freq_size: int = 256):
        super().__init__()

        self.freq_size = freq_size

        self.mlp = nn.Sequential(
            nn.Linear(freq_size, out_channels, bias=True),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def timestep_embedding(self, t: Tensor, max_period=10000) -> Tensor:
        r"""

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            embedding: (b, d)
        """
        
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half).to(t.device)  # (b,)
        args = t[:, None] * freqs[None, :]  # (b, dim/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (b, dim)
        
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        r"""Calculate time embedding.

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            out: (b, d)
        """

        t = self.timestep_embedding(t)
        t = self.mlp(t)
        
        return t


class LabelEmbedder(nn.Module):

    def __init__(self, classes_num: int, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Embedding(classes_num, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: LongTensor) -> Tensor:
        r"""Calculate label embedding.

        Args:
            x: (b,), LongTensor

        Outputs:
            out: (b, d)
        """
        
        return self.mlp(x)


class MlpEmbedder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate MLP embedding.

        Args:
            x: (b, d, ...)

        Outputs:
            out: (b, d, ...)
        """
        
        x = x.transpose(1, -1)  # (b, ..., d)
        x = self.mlp(x)  # (b, ..., d)  let mlp work on d
        x = x.transpose(1, -1)  # (b, d, ...)
        return x
    
class CNNEmbedder(nn.Module):

    def __init__(self, cnn_dim: int, out_channels: int):
        super().__init__()

        self.cnn = Cnn(cnn_dim=cnn_dim)
        in_channels = self.cnn.mlp_in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate CNN embedding. 
        Args:
            x: (b, c, n, n)
        Outputs:
            out: (b, d, ...)
        """
        
        x = self.cnn(x) # (b, c)

        x = self.mlp(x)  # (b, d)  let mlp work on d

        return x
    
class C_AVGEmbedder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate C_AVG embedding.

        Args:
            x: (b, l'd0)

        Outputs:
            out: (b, d) 
        """

        x = self.mlp(x) # (b, d)

        return x
    
class AVGEmbedder(nn.Module):

    def __init__(self, d0: int=6, l_prime: int=20):
        super().__init__()

        self.d0 = d0
        self.l_prime = l_prime

        self.mlp0 = nn.Sequential(
            nn.Linear(2, self.d0),
            nn.SiLU(),
            nn.Linear(self.d0, self.d0, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate C_AVG embedding.

        Args:
            x: (l, 2)

        Outputs:
            out: (l'd0)
        """
        
        x = self.mlp0(x)  # (l, d0) 

        x = x.transpose(0, 1)  # (d0, l)
        pool1d = nn.AdaptiveAvgPool1d(self.l_prime)
        x = pool1d(x)            # (d0, l')
        x = x.transpose(0, 1)  # (l', d0)

        from einops import rearrange
        x = rearrange(x, 'l d -> (l d)')  # (l'd0)

        return x