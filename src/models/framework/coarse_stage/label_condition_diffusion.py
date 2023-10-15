import os
import functools
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
import torch.nn.functional as F
import copy

from .openai_diffusion.diffusion.resample import create_named_schedule_sampler, LossAwareSampler, UniformSampler
from models.module_utils.point_sample_gather import get_fps_point, pc_to_uv
import time
from .base_diffusion import DiffusionModule as BaseModule

class DiffusionModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, fps_pos_normal, shape_cond, data_label):
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)
        model_kwargs = {"fps_cond": fps_pos_normal, "shape_cond": shape_cond, "label_cond": data_label}
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.net,
            x,
            t,
            model_kwargs=model_kwargs,

        )
        losses, pred, target, x_noise = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        return loss, pred, x_noise

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

        shape_cond = torch.cat([cond_pos, cond_normal, mask], dim=1)

        fps_color = batch['input']['fps_color'].to(device=self.device).permute(0, 2, 1)
        fps_pos = batch['input']['fps_points'].to(device=self.device).permute(0, 2, 1)
        fps_normal = batch['input']['fps_normal'].to(device=self.device).permute(0, 2, 1)
        fps_pos_normal = torch.cat([fps_pos, fps_normal], dim=1)

        return fps_color, fps_pos_normal, shape_cond

    def training_step(self, batch: Any, batch_idx: int):
        x, fps_pos_normal, shape_cond = self.step(batch)
        data_label = batch['input']['label'].to(device=self.device)
        loss, pred, x_noise = self.forward(x, fps_pos_normal, shape_cond, data_label)

        return {"loss": loss}, {"image": None}

    def test_step(self, batch: Any, batch_idx: int):
        x, fps_pos_normal, shape_cond = self.step(batch)

        sample_fn = self.diffusion.p_sample_loop

        b, c, n = x.size()
        batch_shape = (b, c, n)

        data_label = torch.randint(low=0, high=40, size=(b,)).to(device=self.device)

        model_kwargs = {"fps_cond": fps_pos_normal, "shape_cond": shape_cond, "label_cond": data_label}
        sample = sample_fn(
            self.net,
            batch_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )

        sample = (1+sample)/2

        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float()

        colors = pc_to_uv(sample.float().permute(0, 2, 1), fps_pos_normal[:, :3, ...].float().permute(0, 2, 1), shape_cond[:, :3, ...].permute(0, 2, 3, 1))
        colors = colors.permute(0, 3, 1, 2) * mask

        obj_c = batch['category']
        obj_name = batch['name']

        point={}
        point['xyz'] = fps_pos_normal[:, :3, ...].float().permute(0, 2, 1)
        point['rgb'] = sample.float().permute(0, 2, 1)

        return {"loss": None}, {"image": colors, "texture_map": colors, "obj_c": obj_c, "obj_name": obj_name, "mask": mask, "self_label": data_label, "point": point}
