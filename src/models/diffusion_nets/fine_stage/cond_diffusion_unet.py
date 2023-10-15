import pdb

from .diffusion_unet_nd import *
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

import clip

class UV_PointNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UV_PointNet, self).__init__()
        self.fc1 = nn.Conv1d(in_ch, 64, 1)
        self.fc2 = nn.Conv1d(64, 128, 1)
        self.fc3 = nn.Conv1d(128, 256, 1)
        self.relu = nn.ReLU()
        self.fc_final = nn.Conv1d(256+128, out_ch, 1)

    def forward(self, x):
        b,c,h,w=x.size()
        x_flat = x.view(b,c,h*w)
        x_flat = self.relu(self.fc1(x_flat))
        x_flat = self.relu(self.fc2(x_flat))
        x_glob_feat = self.fc3(x_flat)
        x_feat = th.max(x_glob_feat, -1, keepdim=True)[0]
        x_feat = th.cat([x_flat, x_feat.repeat(1, 1, h*w)], dim=1)
        x_feat = self.fc_final(x_feat)
        out = x_feat.view(b,-1,h,w)
        return out

class Basic_Cond_Model(UNetModel):
    def __init__(self, in_channels, cond_embed_ch, *args, **kwargs):
        super().__init__(in_channels + cond_embed_ch, *args, **kwargs)
        self.shape_encoder = UV_PointNet(7, cond_embed_ch)

    def forward(self, x, timesteps, cond=None, **kwargs):
        assert cond.size()[1] == 7
        shape_cond = cond

        uv_position = shape_cond[:, :3, ...]
        uv_normal = shape_cond[:, 3:6, ...]
        uv_mask = shape_cond[:, 6:, ...]

        shape_feat = self.shape_encoder(shape_cond)
        x = th.cat([x, shape_feat], dim=1)

        return super().forward(x, timesteps, **kwargs)

class Coarse_Fine_Model(UNetModel):
    def __init__(self, in_channels, cond_embed_ch, *args, **kwargs):
        super().__init__(in_channels + cond_embed_ch + 3, *args, **kwargs)
        self.shape_encoder = UV_PointNet(7, cond_embed_ch)

    def forward(self, x, timesteps, cond=None, **kwargs):
        assert cond.size()[1] == 10
        shape_cond = cond[:, 3:, ...]
        coarse_cond = cond[:, :3, ...]

        uv_position = shape_cond[:, :3, ...]
        uv_normal = shape_cond[:, 3:6, ...]
        uv_mask = shape_cond[:, 6:, ...]

        shape_feat = self.shape_encoder(shape_cond)
        x = th.cat([x, shape_feat, coarse_cond], dim=1)

        return super().forward(x, timesteps, **kwargs)

class Coarse_Fine_Model_Hybrid(UNetModel):
    def __init__(self, in_channels, cond_embed_ch, *args, **kwargs):
        super().__init__(in_channels + cond_embed_ch + 6, *args, **kwargs)
        self.shape_encoder = UV_PointNet(7, cond_embed_ch)

    def forward(self, x, timesteps, cond=None, **kwargs):
        assert cond.size()[1] == 13
        shape_cond = cond[:, 6:, ...]
        coarse_cond = cond[:, :6, ...]

        uv_position = shape_cond[:, :3, ...]
        uv_normal = shape_cond[:, 3:6, ...]
        uv_mask = shape_cond[:, 6:, ...]

        shape_feat = self.shape_encoder(shape_cond)
        x = th.cat([x, shape_feat, coarse_cond], dim=1)

        return super().forward(x, timesteps, **kwargs)

class CondResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels + 3, *args, **kwargs)

    def forward(self, x, timesteps, cond=None, **kwargs):
        x = th.cat([x, cond[:, :3, ...]], dim=1)
        return super().forward(x, timesteps, **kwargs)

class Condition_Coarse_Fine_Model_Hybrid(UNetModel):
    def __init__(self, in_channels, cond_embed_ch, model_channels, *args, **kwargs):
        super().__init__(in_channels + cond_embed_ch + 6, model_channels, *args, **kwargs)
        self.shape_encoder = UV_PointNet(7, cond_embed_ch)
        self.embedding_size = 768
        self.text_embedding = nn.Sequential(
            linear(self.embedding_size, model_channels*4),
            nn.SiLU(),
            linear(model_channels*4, model_channels*4),
        )

    def forward(self, x, timesteps, cond=None, **kwargs):
        text = cond['text']
        cond = cond['other_cond']
        assert cond.size()[1] == 13
        shape_cond = cond[:, 6:, ...]
        coarse_cond = cond[:, :6, ...]

        uv_position = shape_cond[:, :3, ...]
        uv_normal = shape_cond[:, 3:6, ...]
        uv_mask = shape_cond[:, 6:, ...]

        shape_feat = self.shape_encoder(shape_cond)

        x = th.cat([x, shape_feat, coarse_cond], dim=1)


        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        with th.no_grad():
            text_embed = text.float()

        text_embed = self.text_embedding(text_embed)

        emb = emb + text_embed

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)