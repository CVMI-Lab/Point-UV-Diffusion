import os.path
import pdb
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import torch
from typing import List, Optional, Tuple
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.utils import utils
from hydra import compose, initialize
from pytorch_lightning.utilities import rank_zero_only
import random

import torch.multiprocessing as mp
from config_utils import *
from src.metric_evaluation.nv_render import Render
import os
import time
from tqdm import tqdm
import glob
from cleanfid import fid
import logging

def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def evaluate(args: DictConfig) -> Tuple[dict, dict]:
    # initialize device settings
    device = utils.distributed_init(args)
    datamodule = hydra.utils.instantiate(args.datamodule)

    print(args.evaluate_result_folder)

    render_tool = Render()

    # 1. render ground truth images
    split_file = f"{args.datamodule.data_detail.split_files}/{args.datamodule.data_detail.category}/{args.datamodule.data_split.test_split}.lst"
    with open(split_file, 'r') as f:
        split_list = f.readlines()
    gt_dir = f"{args.datamodule.data_detail.uv_folder}/{args.datamodule.data_detail.category}"
    names = [file[:-1] for file in split_list]
    gt_img_save_dir = f"{args.evaluate_result_folder}/evaluation/gt"
    os.makedirs(gt_img_save_dir, exist_ok=True)

    for k, name in tqdm(enumerate(names), total=len(names), desc='render ground-truth', ncols=70):
        file_path = os.path.join(gt_dir, name, 'uv_texture_512.obj')
        img_path = os.path.join(gt_dir, name, 'uv_texture_hom_512.png')
        mesh_file = os.path.join(gt_dir, name)
        try:
            render_tool.render_mesh(mesh_file, file_path, img_path, gt_img_save_dir)
        except Exception as e:
            print(f'Error while processing {file_path}: {e}')

    # 2. render images
    gen_save_dir = f"{args.evaluate_result_folder}/evaluation/generate"
    os.makedirs(gen_save_dir, exist_ok=True)

    # Use glob to find all .obj files in dir and subdirectories
    obj_files = glob.glob(os.path.join(args.evaluate_result_folder, '**/*.obj'), recursive=True)

    for k, file_path in tqdm(enumerate(obj_files), total=len(obj_files), desc='Processing obj files', ncols=70):
        mesh_file, _ = os.path.splitext(file_path)  # Remove .obj extension
        img_path = mesh_file + '.png'  # Change .obj to .png

        try:
            render_tool.render_mesh(mesh_file, file_path, img_path, gen_save_dir)
        except Exception as e:
            print(f'Error while processing {file_path}: {e}')

    # 3. compute metric
    log_path = f"{args.evaluate_result_folder}/evaluation/metric.log"
    set_logging(log_path=log_path)

    print('computing fid ......')
    score_fid = fid.compute_fid(gt_img_save_dir, gen_save_dir)
    print(score_fid)
    score_kid = fid.compute_kid(gt_img_save_dir, gen_save_dir)
    print(score_kid)
    # Write scores to log file
    with open(log_path, 'a') as f:  # 'a' opens the file for appending
        f.write(f'FID Score: {score_fid}\n')
        f.write(f'KID Score: {score_kid}\n')
    logging.warning('score_fid: %f, score_kid: %f' % (score_fid, score_kid))


def get_conf_str(conf_overrides):
    flatten_dicts = flatten_dict(conf_overrides)
    overrides_list = [f"{k}={v}" for k, v in flatten_dicts.items()]

    return overrides_list

def main(config_path="../configs", config_name="test"):
    # Register the new resolver
    OmegaConf.register_new_resolver("dir_resolver", dir_resolver)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
    # set configuration
    cli_args = OmegaConf.to_container(OmegaConf.from_cli(), resolve=True)

    # Remove --local_rank from cli_args and add it back as local_rank
    local_rank = cli_args.pop("--local_rank", None)
    if local_rank is not None:
        cli_args["local_rank"] = local_rank

    overrides_list = get_conf_str(cli_args)

    with initialize(version_base=None, config_path=config_path):
        args = compose(config_name=config_name, overrides=overrides_list)

    evaluate(args)

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    main()
