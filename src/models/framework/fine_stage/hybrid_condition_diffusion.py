import os
import pdb
import random
from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import trimesh
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
from geometry_tools.sample_camera_distribution import *
from geometry_tools.camera import *
from geometry_tools.rasterize_mesh import *
import importlib
from timm.utils import ModelEmaV2
from .render_diffusion import DiffusionModule as BaseModule
from geometry_tools.nvdiffrast.nvdiffrast import torch as dr
from torch_scatter import scatter_mean

class DiffusionModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch: Any):
        x = batch['input']['texture'].to(device=self.device)
        x = x.permute(0, 3, 1, 2).float()
        cond_pos = batch['input']['position'].to(device=self.device)
        b, c, h, w = x.size()
        cond_pos = cond_pos.reshape(b, h, w, c)
        cond_pos = torch.flip(cond_pos, dims=[1]) # important!
        cond_pos = cond_pos.permute(0, 3, 1, 2).float()

        cond_normal = batch['input']['normal'].to(device=self.device)
        b, c, h, w = x.size()
        cond_normal = cond_normal.reshape(b, h, w, c)
        cond_normal = torch.flip(cond_normal, dims=[1])  # important!
        cond_normal = cond_normal.permute(0, 3, 1, 2).float()

        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float()

        coarse_map = batch['input']['coarse_map'].to(device=self.device)
        coarse_map = coarse_map.permute(0, 3, 1, 2).float()

        b, c, h, w = coarse_map.size()
        data_img = coarse_map.reshape(b, c, h * w)
        cluster_labels = batch['input']['cluster_label'].unsqueeze(1).to(device=self.device)
        cluster_labels = cluster_labels.reshape(b, 1, h * w)
        smooth_map = []
        for i in range(b):
            cluster_label = cluster_labels[i:i+1]
            colors = scatter_mean(data_img[i:i+1], cluster_label, dim=-1)
            colors_full = colors[:, :, cluster_label[0][0]]
            colors_full = colors_full.reshape(1, c, h, w)
            smooth_map.append(colors_full)

        smooth_map = torch.cat(smooth_map, dim=0)
        smooth_map = smooth_map * mask


        cond = torch.cat([coarse_map, smooth_map, cond_pos, cond_normal, mask], dim=1)

        return x, cond, smooth_map

    def test_step(self, batch: Any, batch_idx: int):
        x, cond, coarse_map = self.step(batch)

        sample_fn = self.diffusion.p_sample_loop
        #sample_fn = self.diffusion.p_sample_loop_ddim
        #inference_timesteps = 64

        b, c, h, w = x.size()
        batch_shape = (b, c, h, w)

        self.diffusion.net.training = False
        sample = sample_fn(
            self.device,
            batch_shape,
            #timesteps=inference_timesteps,
            cond=cond,
            clip_denoised=False,
            cond_drop_time=1-self.cond_truncated_time,
        )

        mask = cond[:, -1:, ...]
        sample = (1+sample)/2 * mask

        cond_pos = cond[:, 3:6, ...]
        image = torch.cat([sample, (1+coarse_map)/2, cond_pos+0.5], dim=-1)

        obj_c = batch['category']
        obj_name = batch['name']

        return {"loss": None}, {"image": image, "texture_map": sample, "obj_c": obj_c, "obj_name": obj_name, "mask": mask}