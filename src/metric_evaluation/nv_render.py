import os
import pdb
import random
from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import trimesh
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.dirname(__file__))
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
from geometry_tools.sample_camera_distribution import *
from geometry_tools.camera import *
from geometry_tools.rasterize_mesh import *
import importlib
from geometry_tools.nvdiffrast.nvdiffrast import torch as dr
from PIL import Image
import time
import argparse


class Render:
    def __init__(
            self,
            device='cuda',
            view_num=4,
    ):
        super().__init__()
        self.device = device
        self.view_num = view_num

        self.ctx = dr.RasterizeGLContext(device=self.device)
        self.projection_mtx = None
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)
        self.camera = dmtet_camera

    def get_uv_view(self, mesh, camera_mv_bx4x4, resolution=256):
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
        uv_view, _ = interpolate(mesh['v_uv'][None, ...], rast, mesh['f_uv'])
        mask = (torch.sum(rast, dim=-1) > 0)

        return uv_view, mask.float().unsqueeze(1)

    def render_mesh(self, mesh_file, file_path, img_path, save_dir):

        rotation_angle_list = np.random.rand(self.view_num)
        elevation_angle_list = np.random.rand(self.view_num)
        rotation_camera = rotation_angle_list * 360
        elevation_camera = elevation_angle_list * 30

        name = mesh_file.split('/')[-1]
        save_name = os.path.join(save_dir, name + '_3.png')
        if os.path.exists(save_name):
            print('exists')
            return

        cam_mv = align_blender_imgs(rotation_camera, elevation_camera, self.device)
        mesh, tex_map_ = self.load_one_mesh(mesh_file, file_path, img_path)

        for i, cam_single_view in enumerate(cam_mv):
            name = mesh_file.split('/')[-1]
            save_name = os.path.join(save_dir, name + '_%d.png' % i)
            uv_view, mask = self.get_uv_view(mesh, cam_single_view.unsqueeze(0), resolution=512)
            tex_map = torch.flip(tex_map_, dims=[0])

            tex_map = tex_map.unsqueeze(0).contiguous()
            gt_image = dr.texture(tex_map.float(), uv_view).detach()

            gt_image = gt_image.permute(0, 3, 1, 2)

            gt_image = gt_image * mask + (1 - mask) * torch.ones(gt_image.size()).cuda()

            
            torchvision.utils.save_image(gt_image, save_name)

    def load_one_mesh(self, file_name, file_path, img_path):
        tex_map = torch.from_numpy(load_image(img_path)).to('cuda')
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

        return mesh, tex_map


def load_image(path):
    image = Image.open(path)
    image = np.array(image) / 255
    return image


