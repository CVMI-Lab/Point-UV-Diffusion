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
from .base_diffusion import DiffusionModule as BaseModule

from geometry_tools.nvdiffrast.nvdiffrast import torch as dr

class DiffusionModule(BaseModule):
    def __init__(self, render_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_renderer(render_config)

    def build_renderer(self, render_config):
        self.view_num = render_config.view_num
        self.patch_size = render_config.patch_size
        self.render_loss = render_config.render_loss
        self.render_weight = render_config.render_weight
        self.rast_resolution = render_config.rast_resolution
        self.ctx = dr.RasterizeGLContext(device=self.device)
        self.projection_mtx = None
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)
        self.camera = dmtet_camera

    def training_step(self, batch: Any, batch_idx: int):
        x, cond, coarse_map = self.step(batch)
        loss, pred, x_noise = self.forward(x, cond)

        mask = batch['input']['mask'].to(device=self.device)
        mask = mask.unsqueeze(1).float()

        pred = (1+pred)/2
        x = (1+x)/2
        dilated_pred = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1, dilation=1)
        dilated_gt = F.max_pool2d(x, kernel_size=3, stride=1, padding=1, dilation=1)

        dilated_pred = dilated_pred * (1-mask) + mask * pred
        dilated_gt = dilated_gt * (1 - mask) + mask * x
        render_loss_list = []
        pred_images = []
        gt_images = []

        mesh_file = batch['mesh_file']
        camera = batch.get('camera')
        if self.render_weight != 0:
            for b, one_file in enumerate(mesh_file):
                if camera != None:
                    rotation_camera = camera['rotation'][b]
                    elevation_camera = camera['elevation'][b]

                    indices = np.random.randint(rotation_camera.shape[0], size=self.view_num)
                    rotation_camera = rotation_camera[indices]
                    elevation_camera = elevation_camera[indices]
                else:
                    rotation_angle_list = np.random.rand(self.view_num)
                    elevation_angle_list = np.random.rand(self.view_num)
                    rotation_camera = rotation_angle_list * 360
                    elevation_camera = elevation_angle_list * 30

                cam_mv = align_blender_imgs(rotation_camera, elevation_camera, self.device)
                mesh = self.load_one_mesh(one_file)

                for i, cam_single_view in enumerate(cam_mv):
                    render_loss_one, pred_image, gt_image = self.render_loss_fn(mesh, cam_single_view.unsqueeze(0).float(),
                                                                                dilated_pred[b], dilated_gt[b])
                    render_loss_list.append(render_loss_one)
                    pred_images.append(pred_image)
                    gt_images.append(gt_image)

            render_loss = torch.stack(render_loss_list, dim=0)
            render_loss = self.render_weight * torch.mean(render_loss)

            image = torch.cat([pred, x, (x_noise + 1) / 2.0, (coarse_map + 1) / 2.0, dilated_gt], dim=-1)

            render_image = torch.cat(pred_images + gt_images, dim=-1)
            render_image = render_image

            loss = loss + render_loss
            return {"loss": loss, "render_loss": render_loss}, {"image": image, "render_image": render_image}
        else:
            image = torch.cat([pred, x, (x_noise + 1) / 2.0, (coarse_map + 1) / 2.0, dilated_gt], dim=-1)
            return {"loss": loss, "render_loss": 0}, {"image": image}


    def render_loss_fn(self, mesh, camera_mv_bx4x4, pred_tex_map, gt_tex_map):
        # step1--step3: do rasterization and interpolation
        uv_view = self.get_uv_view(mesh, camera_mv_bx4x4, resolution=self.rast_resolution, patch=self.patch_size)
        # step4: sampling the texture image
        pred_tex_map = torch.flip(pred_tex_map, dims=[1])
        gt_tex_map = torch.flip(gt_tex_map, dims=[1])

        pred_tex_map = pred_tex_map.permute(1, 2, 0).unsqueeze(0).contiguous()
        gt_tex_map = gt_tex_map.permute(1, 2, 0).unsqueeze(0).contiguous()
        pred_image = dr.texture(pred_tex_map, uv_view)
        gt_image = dr.texture(gt_tex_map, uv_view).detach()

        pred_image = pred_image.permute(0, 3, 1, 2)
        gt_image = gt_image.permute(0, 3, 1, 2)

        render_loss = self.render_loss(pred_image, gt_image)

        return render_loss, pred_image, gt_image

    def select_patch(self, rast, patch_size):
        B, H, W, C = rast.size()
        rast_mask = rast[:, :, :, 3]
        rast_patchs = []
        for b in range(B):
            h_mask = torch.mean(rast_mask[b:b + 1], dim=2).cpu().numpy()
            w_mask = torch.mean(rast_mask[b:b + 1], dim=1).cpu().numpy()
            h_idx = np.where(h_mask > 0)
            w_idx = np.where(w_mask > 0)
            h_min, h_max = h_idx[1][0], h_idx[1][-1]
            w_min, w_max = w_idx[1][0], w_idx[1][-1]
            h_min = min(h_min, H - patch_size)
            w_min = min(w_min, W - patch_size)
            h_max = max(h_max, h_min + patch_size)
            w_max = max(w_max, w_min + patch_size)
            h1 = random.randint(h_min, h_max - patch_size)
            w1 = random.randint(w_min, w_max - patch_size)
            rast_patch = rast[b:b + 1, h1:h1 + patch_size, w1:w1 + patch_size, :]
            rast_patchs.append(rast_patch)

        return torch.cat(rast_patchs, dim=0)

    def get_uv_view(self, mesh, camera_mv_bx4x4, resolution=1024, patch=False):
        '''
        :param mesh:
               mesh['v'] = vertices
                mesh['f'] = faces
                mesh['v_uv'] = uv_vertices
                mesh['f_uv'] = uv_faces

        :param camera_mv_bx4x4:
        :return:
        '''
        # step1: transform to clip space
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(
            camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh['v'], mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
        # step2: Rasterization
        rast, _ = dr.rasterize(self.ctx, v_pos_clip, mesh['f'], (resolution, resolution))
        # step3: Interpolation
        if not patch:
            uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast, mesh['f_uv'])
        else:
            rast_patch = self.select_patch(rast, self.patch_size).contiguous()
            uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast_patch, mesh['f_uv'])

        return uv_view

    def load_one_mesh(self, file_path):
        vertex_data = []
        face_data = []
        uv_vertex_data = []
        uv_face_data = []
        for line in open(file_path, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                vertex_data.append(v)

            elif values[0] == 'vt':
                vt = list(map(float, values[1:3]))
                uv_vertex_data.append(vt)

            elif values[0] == 'f':
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                uv_f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                uv_face_data.append(uv_f)

        vertices = torch.from_numpy(np.array(vertex_data)).to(self.device)
        faces = torch.from_numpy(np.array(face_data)).to(self.device)
        uv_vertices = torch.from_numpy(np.array(uv_vertex_data)).to(self.device)
        uv_faces = torch.from_numpy(np.array(uv_face_data)).to(self.device)

        mesh = {}
        mesh['v'] = vertices.float()
        mesh['f'] = faces.int() - 1
        mesh['v_uv'] = uv_vertices.float()
        mesh['f_uv'] = uv_faces.int() - 1

        return mesh