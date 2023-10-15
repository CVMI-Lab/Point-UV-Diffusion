import pdb
import time
import glob
import random
from PIL import Image
import numpy as np
import trimesh
import os
import logging
from torch.utils import data
import yaml
import h5py
import torch.nn.functional as F
import torch_cluster
import json
logger = logging.getLogger(__name__)
from skimage import measure, color

class CameraField(object):
    def __init__(self, transform=None):
        self.transform = transform

    def load(self, camera_path):
        rotation_camera = np.load(os.path.join(camera_path, 'rotation.npy'))
        elevation_camera = np.load(os.path.join(camera_path, 'elevation.npy'))
        assert len(rotation_camera) == len(elevation_camera)

        data_out = {}

        data_out['rotation'] = rotation_camera
        data_out['elevation'] = elevation_camera

        if self.transform is not None:
            data_out = self.transform(data_out)

        return data_out

def get_camera_field(args):
    field = CameraField()
    return field

class MeshField(object):
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path):
        file_path = os.path.join(model_path, self.file_name)
        # mesh length is not the same, get problem with pytorch dataloader
        return file_path


def get_mesh_field(args):
    field = MeshField(
        args.mesh_file,
    )
    return field

class uvField(object):
    def __init__(self, mask_name, texture_name, position_name, transform=None):
        self.mask_name = mask_name
        self.texture_name = texture_name
        self.position_name = position_name
        self.transform = transform

    def load(self, uv_path):
        uv_mask_file = os.path.join(uv_path, self.mask_name)
        uv_texture_file = os.path.join(uv_path, self.texture_name)

        uv_pos_file = os.path.join(uv_path, self.position_name)
        pointcloud_dict = np.load(uv_pos_file)
        points = pointcloud_dict['points'].astype(np.float32)
        normal = pointcloud_dict['normals'].astype(np.float32)

        normal[np.isnan(normal)] = 0

        uv_mask_image = Image.open(uv_mask_file)
        uv_mask_image = np.array(uv_mask_image) / 255

        uv_texture_image = Image.open(uv_texture_file)
        uv_texture_image = np.array(uv_texture_image) / 255

        #normalize to [-1, 1]
        uv_texture_image = uv_texture_image*2-1

        cluster_label = measure.label(uv_mask_image, connectivity=1)

        data_out = {}

        data_out['texture'] = uv_texture_image
        data_out['mask'] = uv_mask_image
        data_out['position'] = points
        data_out['normal'] = normal
        data_out['cluster_label'] = cluster_label

        if self.transform is not None:
            data_out = self.transform(data_out)

        return data_out

def get_uv_field(args):
    field = uvField(mask_name=args.mask_file,
                    texture_name=args.texture_file,
                    position_name=args.position_file,
                    )
    return field

def get_coarse_map(file_path):
    uv_texture_image = Image.open(file_path)
    uv_texture_image = np.array(uv_texture_image) / 255

    # normalize to [-1, 1]
    uv_texture_image = uv_texture_image * 2 - 1

    return uv_texture_image

def get_pca_info(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    #pdb.set_trace()
    label = np.array(data['label']).astype(np.int)
    weight = np.array(data['weight']).astype(np.float32)

    return label, weight

def get_fps_point_info(file_path):
    pointcloud_dict = np.load(file_path)
    points = pointcloud_dict['points'].astype(np.float32)
    normal = pointcloud_dict['normal'].astype(np.float32)
    color = pointcloud_dict['color'].astype(np.float32)

    normal[np.isnan(normal)] = 0
    color = color*2-1

    return color, points, normal

class Dataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, split, args, mode):
        # Attributes
        self.mode = mode
        self.split = split
        self.model_folder = args.model_folder
        self.uv_folder = args.uv_folder
        self.uv_field = get_uv_field(args)
        self.coarse_point_folder = args.coarse_point_folder
        self.pca_folder = args.pca_folder

        self.camera_field = get_camera_field(args)
        self.render_folder = getattr(args, 'render_folder', None)  # returns None if 'render_folder' attribute does not exist
        self.mesh_field = get_mesh_field(args)

        self.args = args
        self.get_all_models(args)

    def get_all_models(self, args):
        split = self.split
        category = args.category
        split_files = args.split_files

        # Get all models
        self.models = []

        subpath = os.path.join(self.model_folder, category)
        if not os.path.isdir(subpath):
            logger.warning('Category %s does not exist in dataset.' % category)
        if split == 'all':
            self.models += [
                {'category': category, 'model': m} for m in
                [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '')]
            ]
        else:
            split_file = os.path.join(split_files, category, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.models += [
                {'category': category, 'model': m}
                for m in models_c
            ]
        if self.mode == 'test' and args.test_samples:
            self.models = self.models[:args.test_samples]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        uv_path = os.path.join(self.uv_folder, category, model)
        mesh_path = os.path.join(self.uv_folder, category, model)

        data = {}

        uv_data = self.uv_field.load(uv_path)

        mesh_file = self.mesh_field.load(mesh_path)

        coarse_point_folder = self.coarse_point_folder
        file_path = os.path.join(coarse_point_folder, model + '.npz')
        color, points, normal = get_fps_point_info(file_path)
        uv_data['fps_color'] = color
        uv_data['fps_points'] = points
        uv_data['fps_normal'] = normal

        if self.mode == 'train':
            pca_folder = self.pca_folder
            file_path = os.path.join(pca_folder, model + '.json')
            label, weight = get_pca_info(file_path)
            uv_data['label'] = label
            uv_data['weight'] = weight

        data['input'] = uv_data
        data['mesh_file'] = mesh_file
        data['name'] = model
        data['category'] = category

        if self.render_folder is not None:
            camera_path = os.path.join(self.render_folder, 'camera', category, model)
            camera = self.camera_field.load(camera_path)
            data['camera'] = camera

        return data