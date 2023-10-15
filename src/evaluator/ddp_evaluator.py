import math
import os
import sys
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
import pdb
from src.utils import utils
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import torchvision
import numpy as np
import PIL
import cv2
import open3d as o3d

def save_pointcloud(ply_filename, xyz: torch.Tensor, color: torch.Tensor):
    xyz = xyz.view(-1, 3).cpu().numpy()
    color = color.view(-1, 3).cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(ply_filename, pcd)

    return xyz, color

class MyBaseEvaluator:
    def __init__(
        self,
        distributed=True,
        logger=None,
        device='cuda',
        ckpt_resume=False,
        local_rank=-1,
    ):
        super().__init__()

        self.distributed = distributed
        self.device = device
        self.logger = logger
        self.ckpt_resume = ckpt_resume
        self.local_rank = local_rank

    @torch.no_grad()
    def test(self, datamodule, modelmodule):
        modelmodule.net.to(self.device)
        checkpoint = torch.load(self.ckpt_resume, map_location='cpu')
        if 'model_ema' not in checkpoint.keys():
            modelmodule.net.load_state_dict(checkpoint['model'])
        else:
            print('loading ema model')
            modelmodule.net.load_state_dict(checkpoint['model_ema'])
        modelmodule.net = torch.nn.parallel.DistributedDataParallel(modelmodule.net,
                                                                    device_ids=[self.local_rank],
                                                                    find_unused_parameters=False)
        modelmodule.net.eval()
        data_loader_test = datamodule.test_dataloader()


        tbar = tqdm(data_loader_test)
        save_dir = os.path.join(self.logger.log_dir, 'rank%s' % self.local_rank)
        os.makedirs(save_dir, exist_ok=True)
        obj_dir = datamodule.data_detail.uv_folder
        for batch_idx, batch in enumerate(tbar):
            batch_stat, batch_media = modelmodule.test_step(batch, batch_idx)
            images = batch_media['image']
            texture_maps = batch_media["texture_map"]
            masks = batch_media["mask"]

            _, _, res, _ = texture_maps.size()
            for i in range(images.size()[0]):
                obj_c = batch_media['obj_c'][i]
                obj_name = batch_media['obj_name'][i]
                obj_file = os.path.join(obj_dir, obj_c, obj_name, 'uv_texture_%d.obj' %res)

                texture_map = texture_maps[i:i+1]
                mask = masks[i:i+1]

                self.save_texture_model(obj_file, obj_name, save_dir, texture_map, mask)

                image = images[i:i+1]
                filename = os.path.join(save_dir, '%s_cat.jpg' % obj_name)
                torchvision.utils.save_image(image.detach().cpu(), filename)

                if "self_label" in batch_media.keys():
                    filename = os.path.join(save_dir, '%s.txt' % obj_name)
                    with open(filename, 'w') as f:
                        data_label = batch_media['self_label'][i]
                        f.write('%s' %data_label)
                if "point" in batch_media.keys():
                    filename = os.path.join(save_dir, '%s_fps.ply' % obj_name)
                    point = batch_media['point']
                    xyz = point['xyz'][i]
                    rgb = point['rgb'][i]
                    save_pointcloud(filename, xyz, rgb)

    def save_texture_model(self, obj_file, obj_name, save_dir, texture_map, mask):
        # Modify and save .obj file
        with open(obj_file, 'r+') as f:
            flist = f.readlines()
        flist[0] = f'mtllib {obj_name}.mtl\n'
        with open(os.path.join(save_dir, f'{obj_name}.obj'), 'w+') as f:
            f.writelines(flist)

        # Create and save .mtl file
        with open(os.path.join(save_dir, f'{obj_name}.mtl'), 'w') as fid:
            fid.write(f'newmtl material_0\nKd 1 1 1\nKa 0 0 0\nKs 0.4 0.4 0.4\nNs 10\nillum 2\nmap_Kd {obj_name}.png')

        texturename = os.path.join(save_dir, obj_name + '.png')
        texturename_ori = os.path.join(save_dir, obj_name + '_ori.png')

        color_map = texture_map.permute(0, 2, 3, 1)[0]

        img = np.asarray(color_map.data.cpu().numpy(), dtype=np.float32)
        img = img * 255
        img = img.clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(np.ascontiguousarray(img), 'RGB').save(texturename_ori)

        mask = mask.permute(0, 2, 3, 1)[0]
        mask = np.asarray(mask.data.cpu().numpy(), dtype=np.float32)

        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        hom_img = img * mask + dilate_img * (1 - mask)
        hom_img = hom_img.clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(np.ascontiguousarray(hom_img), 'RGB').save(texturename)
