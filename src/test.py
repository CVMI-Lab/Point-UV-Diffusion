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


def test(args: DictConfig) -> Tuple[dict, dict]:
    # initialize device settings
    device = utils.distributed_init(args['test'])
    datamodule = hydra.utils.instantiate(args['test'].datamodule)
    modelmodule = hydra.utils.instantiate(args['train'].model, device=device)

    evaluator: Evaluator = hydra.utils.instantiate(
        args['test'].evaluator,
        ckpt_resume=args['test'].ckpt_resume,
        device=device,
        local_rank=args['test'].local_rank
    )
    save_yaml_path = os.path.join(evaluator.logger.log_dir, 'config.yaml')
    save_config(args, save_yaml_path)

    if args.get("test"):
        evaluator.test(modelmodule=modelmodule, datamodule=datamodule)

def get_conf_str(conf_overrides):
    flatten_dicts = flatten_dict(conf_overrides)
    overrides_list = [f"{k}={v}" for k, v in flatten_dicts.items() if k != 'config_load']
    config_path_train = flatten_dicts.get('config_load', None)
    ckpt_path_train = flatten_dicts.get('ckpt_name', None)

    return overrides_list, config_path_train, ckpt_path_train

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

    overrides_list, config_path_train, ckpt_path_train = get_conf_str(cli_args)

    parts = ckpt_path_train.split('/')
    desired_parts = parts[-5:-2]
    extracted_string = '/'.join(desired_parts)
    overrides_list.append('exp_name=%s' % extracted_string)

    with initialize(version_base=None, config_path=config_path):
        args_test = compose(config_name=config_name, overrides=overrides_list)

    if config_path_train is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path_train_rel = os.path.relpath(config_path_train, start=script_dir)
        config_path_dir, config_file_name = os.path.split(config_path_train_rel)
        config_name = os.path.splitext(config_file_name)[0]
        with initialize(version_base=None, config_path=config_path_dir):
            args_train = compose(config_name=config_name)
    else:
        args_train = args_test

    args_test.ckpt_resume = args_test.ckpt_name
    args = {'test': args_test, 'train': args_train}

    # change test batch:
    args_test.datamodule.batch_size = args_test.test_batch_size

    test(args)

if __name__ == "__main__":
    #mp.set_start_method('spawn')
    main()
