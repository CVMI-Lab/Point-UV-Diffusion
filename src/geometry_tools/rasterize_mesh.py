# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import torch.nn.functional as F
from .nvdiffrast.nvdiffrast import torch as dr
from .camera import *
import pdb
import numpy as np
import trimesh
import torchvision
import json
from .sample_camera_distribution import *
import os

_FG_LUT = None


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')

def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

class Renderer():
    def __init__(self):
        pass

    def forward(self):
        pass

class NeuralRender(Renderer):
    def __init__(self, device='cuda', camera_model=None):
        super(NeuralRender, self).__init__()
        self.device = device
        self.ctx = None
        self.projection_mtx = None
        self.camera = camera_model

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask
        #pdb.set_trace()
        if self.ctx is None:
            self.ctx = dr.RasterizeGLContext(device=device)
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd, v_pos], dim=-1)  # Concatenate the pos  compute the supervision

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth

def render_mesh(renderer, mesh_v_nx3, mesh_f_fx3, camera_mv_bx4x4, resolution=256, hierarchical_mask=False):
    return_value = dict()
    tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth = renderer.render_mesh(
        mesh_v_nx3.unsqueeze(dim=0),
        mesh_f_fx3.int(),
        camera_mv_bx4x4,
        mesh_v_nx3.unsqueeze(dim=0),
        resolution=resolution,
        device=mesh_v_nx3.device,
        hierarchical_mask=hierarchical_mask
    )

    return_value['tex_pos'] = tex_pos
    return_value['mask'] = mask
    return_value['hard_mask'] = hard_mask
    return_value['rast'] = rast
    return_value['v_pos_clip'] = v_pos_clip
    return_value['mask_pyramid'] = mask_pyramid
    return_value['depth'] = depth
    #pdb.set_trace()

    return return_value

def normalize_mesh(mesh, mesh_scale=0.7):
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    assert center.all() == 0
    scale = mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    mesh.vertices = vertices

    return mesh

def align_blender_imgs(rotation_camera, elevation_camera, device):
    assert len(rotation_camera) == len(elevation_camera)
    cam_mv = []
    for img_idx, frame in enumerate(rotation_camera):
        # see GET3D-->dataset.py--L312
        theta = rotation_camera[img_idx] / 180 * np.pi
        phi = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
        theta = torch.from_numpy(np.array(theta))
        phi=torch.from_numpy(np.array(phi))
        # see sample_camera_distribution.py--L107-L117
        compute_theta = -theta - 0.5 * math.pi
        output_points = torch.zeros((1, 3), device=device)
        sample_r = 1.2
        output_points[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(compute_theta)
        output_points[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(compute_theta)
        output_points[:, 1:2] = sample_r * torch.cos(phi)
        # see sample_camera_distribution.py--L78
        forward_vector = normalize_vecs(output_points)
        cam_pos = create_my_world2cam_matrix(forward_vector, output_points, device=device)

        #cam_pos = torch.from_numpy(cam_pos)
        cam_mv.append(cam_pos)
    cam_mv = torch.cat(cam_mv, dim=0)
    return cam_mv

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Camera definition, we follow the definition from Blender (check the render_shapenet_data/render_shapenet.py for more details)
    fovy = np.arctan(32 / 2 / 35) * 2
    fovyangle = fovy / np.pi * 180.0
    dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=device)
    # Renderer we used.
    dmtet_renderer = NeuralRender(device, camera_model=dmtet_camera)
    # load mesh
    file_path = './test_datas/03001627/1006be65e7bc937e9141f9b58470d646/water_model.obj'
    mesh = trimesh.load(file_path, force='mesh')
    mesh = normalize_mesh(mesh)

    verts, faces = torch.from_numpy(np.array(mesh.vertices)), torch.from_numpy(np.array(mesh.faces))
    verts = verts.to(device).squeeze(0).float()
    faces = faces.to(device).squeeze(0).float()
    # camera views for render
    data_camera_mode = 'shapenet_chair'
    '''
    # random sample views
    batch_size = 1
    n_views = 4
    campos, cam_mv, rotation_angle, elevation_angle, sample_r = generate_random_camera(
        data_camera_mode, device, batch_size, n_views=n_views)
    cam_mv = cam_mv.squeeze(0).float()
    # pdb.set_trace()
    '''
    # sample views aligned to blender imgs
    view_path = './test_datas/03001627/1006be65e7bc937e9141f9b58470d646/blender_imgs'
    cam_mv = align_blender_imgs(view_path, device).float()

    #pdb.set_trace()
    for i, cam_single_view in enumerate(cam_mv):
        return_value = render_mesh(dmtet_renderer, verts, faces, cam_single_view.unsqueeze(0),
                                   resolution=1024)
        depth = return_value['depth']
        pc = return_value['tex_pos']
        torchvision.utils.save_image(depth.permute(0, 3, 1, 2), 'output/depth_%s.png' %i, normalize=True)
        save_pointcloud('output/pc_%s.ply' %i, pc)
        #pdb.set_trace()
    pc, _ = trimesh.sample.sample_surface(mesh, count=20000)
    pc = torch.from_numpy(pc)
    save_pointcloud('output/pc.ply', pc)