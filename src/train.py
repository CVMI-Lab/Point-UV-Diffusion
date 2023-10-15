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
import sys
from config_utils import *

def train(args: DictConfig) -> Tuple[dict, dict]:
    # initialize device settings
    device = utils.distributed_init(args)
    datamodule = hydra.utils.instantiate(args.datamodule)
    modelmodule = hydra.utils.instantiate(args.model, local_rank=args.local_rank, device=device)

    modelmodule.net = torch.nn.parallel.DistributedDataParallel(modelmodule.net, device_ids=[args.local_rank],
                                                         find_unused_parameters=True)
    modelmodule.net_without_ddp = modelmodule.net.module

    trainer: Trainer = hydra.utils.instantiate(
        args.trainer,
        ckpt_resume=args.ckpt_resume,
        device=device,
        modelmodule=modelmodule,
        )
    command_line = get_command()
    trainer.logger.save_command(command_line)

    save_yaml_path = os.path.join(trainer.logger.log_dir, 'config.yaml')
    save_config(args, save_yaml_path)
    if args.get("train"):
        trainer.train(modelmodule=modelmodule, datamodule=datamodule, ckpt_path=None)


def main(config_path="../configs", config_name="train"):
    # Register the new resolver
    OmegaConf.register_new_resolver("dir_resolver", dir_resolver)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
    # set configuration
    cli_args = OmegaConf.from_cli()
    # Add additional override for 'task_name'
    task_name = cli_args.get("experiment")
    if task_name is not None:
        cli_args["task_name"] = task_name

    # Remove --local_rank from cli_args and add it back as local_rank
    cli_args_dict = OmegaConf.to_container(cli_args, resolve=True)
    cli_args_dict['local_rank'] = cli_args_dict.pop("--local_rank", None)
    flat_cli_args = flatten_dict(cli_args_dict)

    # Create the overrides list
    overrides_list = [f'{k}={v}' for k, v in flat_cli_args.items()]
    with initialize(version_base=None, config_path=config_path, job_name="test_app"):
        args = compose(config_name=config_name, overrides=overrides_list)

    # resume the training
    config_load = args.config_load
    if config_load:
        args = OmegaConf.load(config_load+'/config.yaml')
        args.ckpt_resume = os.path.join(config_load, 'ckpt')
        args.config_load = config_load

    train(args)

if __name__ == "__main__":
    main()
