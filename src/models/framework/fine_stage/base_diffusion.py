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

class DiffusionModule:
    def __init__(
        self,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device='cuda',
        model_ema_decay=None,
        local_rank=-1,
        cond_truncated_time=0.4,
    ):
        super().__init__()
        self.device = device
        self.diffusion = diffusion
        self.net = self.diffusion.net
        self.net.to(device)
        self.net_ema = None
        if model_ema_decay:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.net_ema = ModelEmaV2(
                self.net,
                decay=model_ema_decay,
            )
            print("Using EMA with decay = %.8f" % model_ema_decay)

        self.net_without_ddp = self.net

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        self.optimizer = optimizer(params=self.net.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.cond_truncated_time = cond_truncated_time


    def forward(self, x: torch.Tensor, cond_pos: torch.Tensor):
        loss, pred, x_noise = self.diffusion(x, cond_pos)
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

        coarse_map = batch['input']['coarse_map'].to(device=self.device)
        coarse_map = coarse_map.permute(0, 3, 1, 2).float()

        cond = torch.cat([coarse_map, cond_pos, cond_normal, mask], dim=1)

        return x, cond, coarse_map

    def training_step(self, batch: Any, batch_idx: int):
        x, cond, coarse_map = self.step(batch)
        loss, pred, x_noise = self.forward(x, cond)

        image = torch.cat([pred, x, x_noise, coarse_map], dim=-1)
        image = (image+1)/2.0
        return {"loss": loss}, {"image": image}

    def test_step(self, batch: Any, batch_idx: int):
        x, cond, coarse_map = self.step(batch)

        sample_fn = self.diffusion.p_sample_loop

        b, c, h, w = x.size()
        batch_shape = (b, c, h, w)

        self.diffusion.net.training = False
        sample = sample_fn(
            self.device,
            batch_shape,
            cond=cond,
            clip_denoised=False,
        )

        mask = cond[:, -1:, ...]
        sample = (1+sample)/2 * mask

        cond_pos = cond[:, 3:6, ...]
        image = torch.cat([sample, (1+coarse_map)/2, cond_pos+0.5], dim=-1)

        obj_c = batch['category']
        obj_name = batch['name']

        return {"loss": None}, {"image": image, "texture_map": sample, "obj_c": obj_c, "obj_name": obj_name, "mask": mask}
