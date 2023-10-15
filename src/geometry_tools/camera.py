# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
from torch import nn
from .sample_camera_distribution import sample_camera, create_camera_from_angle

class Camera(nn.Module):
    def __init__(self):
        super(Camera, self).__init__()
        pass

def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / x, 0, 0, 0],
         [0, n / -x, 0, 0],
         [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)


class PerspectiveCamera(Camera):
    def __init__(self, fovy=49.0, device='cuda'):
        super(PerspectiveCamera, self).__init__()
        self.device = device
        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        self.proj_mtx = torch.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1))

    def project(self, points_bxnx4):
        out = torch.matmul(
            points_bxnx4,
            torch.transpose(self.proj_mtx.to(points_bxnx4.device).unsqueeze(dim=0), 1, 2))
        return out

def generate_random_camera(data_camera_mode, device, batch_size, n_views=2):
    '''
    Sample a random camera from the camera distribution during training
    :param batch_size: batch size for the generator
    :param n_views: number of views for each shape within a batch
    :return:
    '''
    sample_r = None
    world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(
        data_camera_mode, batch_size * n_views, device)
    mv_batch = world2cam_matrix
    campos = camera_origin
    return campos.reshape(batch_size, n_views, 3), mv_batch.reshape(batch_size, n_views, 4, 4), \
           rotation_angle, elevation_angle, sample_r
