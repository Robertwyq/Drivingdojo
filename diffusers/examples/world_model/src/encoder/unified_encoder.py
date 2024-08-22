import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from .convnext import convnext_tiny
import torch.nn.functional as F
from einops import rearrange, repeat
import random

class EgoCondEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, out_dim=1024):
        super().__init__()
        self.out_dim = out_dim

        self.linears = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU())
    
    def forward(self, ego):
        ego_latents = self.linears(ego) + self.shortcut(ego)
        return ego_latents
