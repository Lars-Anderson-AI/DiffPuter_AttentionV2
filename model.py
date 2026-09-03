from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from diffusion_utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        
        # 1. Feature Tokenizer: independently map each of the d_in features to d_model
        # We use a weight matrix of [d_in, d_model] and bias [d_in, d_model]
        self.feature_weights = nn.Parameter(torch.randn(d_in, d_model) * 0.02)
        self.feature_biases = nn.Parameter(torch.zeros(d_in, d_model))

        # 2. Time Embedding layers
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_model) # Project time to d_model so it can be added to tokens
        )

        # 3. Transformer Encoder (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Detokenizer + Final MLP
        # Flatten (B, d_in, d_model) to (B, d_in * d_model) and map to original d_in
        # We project to dim_t first to maintain network capacity similar to original MLP
        self.out_proj = nn.Linear(d_in * d_model, dim_t)
        self.final_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, noise_labels, class_labels=None):
        B = x.shape[0]
        
        # A. Hitung Waktu (Time Embedding) terlebih dahulu
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb) # Shape: (B, d_model)
        
        # B. Feature Tokenization (Mencegah mixing fitur)
        # x is (B, d_in) -> x.unsqueeze(-1) is (B, d_in, 1)
        # feature_weights is (d_in, d_model)
        # element-wise multiplication leverages broadcasting -> (B, d_in, d_model)
        h_tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases
        
        # C. Gabungkan waktu SEBELUM attention!
        # Tambahkan emb (B, d_model) ke semua d_in tokens (B, d_in, d_model)
        h_tokens = h_tokens + emb.unsqueeze(1)
        
        # D. Lewatkan ke Transformer Block (Self-Attention)
        # Perhatian dihitung SECARA SPESIFIK antar kolom/fitur
        h_tokens = self.transformer(h_tokens)
        
        # E. Kembalikan ke bentuk asal dan teruskan ke MLP
        h_flat = h_tokens.reshape(B, -1) # Shape: (B, d_in * d_model)
        h = self.out_proj(h_flat) # Shape: (B, dim_t)
        out = self.final_mlp(h)   # Shape: (B, d_in)
        
        return out


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):

        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()
