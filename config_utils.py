from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import sys

def dir_resolver(include_timestamp, dir_name_with_timestamp, dir_name_without_timestamp):
    return dir_name_with_timestamp if include_timestamp else dir_name_without_timestamp

@rank_zero_only
def save_config(args, path):
    OmegaConf.save(config=args, f=path)

@rank_zero_only
def get_command():
    n = len(sys.argv)
    str = sys.argv[0]
    str += "\n"
    for i in range(1, n):
        str += sys.argv[i]
        str += "\n"

    return str


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)