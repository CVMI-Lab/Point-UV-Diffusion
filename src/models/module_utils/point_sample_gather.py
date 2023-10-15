from torch_cluster import fps, knn
import numpy as np
import scipy.ndimage as ndi
from skimage import measure,color
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import numpy as np
import cv2
from skimage import morphology
from torch_scatter import scatter_mean
import torch
import os
import torchvision
from torch_geometric.nn.unpool import knn_interpolate

def get_fps_point(x, x_pos, fps_num):
    '''
    :param x: Bx3xHxW
    :param x_pos: Bx6(w.normal)xHxW: normal+pos
    :param x_mask: Bx1xHxW
    :return:
    '''
    B = x.size()[0]
    pc_infos = []
    for b in range(B):
        pc = x_pos[b].permute(1, 2, 0)
        color = x[b].permute(1, 2, 0)
        H, W, C = pc.shape
        N = H* W
        pc = pc.view(N, C)
        color = color.view(N, 3)
        ratio = (fps_num + 1) / N
        # idx: get the farthest points in each mini-batch
        idxs = fps(pc[:, -3:], ratio=ratio)  # 0.0625 (2048x0.0625=128)
        idx = idxs[:fps_num]
        fps_pc = pc[idx]
        fps_color = color[idx]
        pc_info = torch.cat([fps_color, fps_pc], dim=-1)
        pc_infos.append(pc_info)
    pc_infos = torch.stack(pc_infos, dim=0)

    return pc_infos

def pc_to_uv(pc_feature_s, pc_pos_s, uv_pos_s):
    '''
    does not support batch processing!!
    '''
    features = []
    for b in range(pc_pos_s.size()[0]):
        pc_feature, pc_pos, uv_pos = pc_feature_s[b], pc_pos_s[b], uv_pos_s[b]
        h, w, _ = uv_pos.size()
        uv_pos = uv_pos.reshape(h*w, 3)

        uv_scatter_feature = knn_interpolate(pc_feature, pc_pos, uv_pos, k=3)
        n, c = pc_feature.size()

        uv_scatter_feature = uv_scatter_feature.reshape(h, w, c)
        features.append(uv_scatter_feature)
    features = torch.stack(features, dim=0)
    return features