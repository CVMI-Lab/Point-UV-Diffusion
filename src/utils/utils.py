from src.utils import device_utils
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
import os
from timm.utils import get_state_dict
import pdb
import io
from timm.utils import ModelEmaV2


def distributed_init(args):
    '''
    function for initializing settings --by xinyu
    :return: device
    '''
    device_utils.init_distributed_mode(args)
    device = torch.device(args.local_rank)

    # fix the seed for reproducibility
    if args.get("seed"):
        pl.seed_everything(args.seed, workers=True)

    return device

@rank_zero_only
def save_model(output_dir, epoch, step, model, model_without_ddp, optimizer, model_ema=None, epoch_name=None, save_meta=False):
    if epoch_name is None:
        epoch_name = str(epoch)

    checkpoint_path = os.path.join(output_dir, 'checkpoint-%s.pth' % epoch_name)
    if save_meta:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)
    else:
        to_save = {}
        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)
        else:
            to_save['model'] = model_without_ddp.state_dict()

    torch.save(to_save, checkpoint_path)

def auto_load_model(output_dir, model, model_without_ddp, optimizer, model_ema=None):

    # deepspeed, only support '--auto_resume'.
    import glob
    latest_path = os.path.join(output_dir, 'checkpoint-latest.pth')
    if os.path.exists(latest_path):
        resume = latest_path
    else:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
    print("Auto resume checkpoint: %s" % resume)
    checkpoint = torch.load(resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'step' in checkpoint:
            step = checkpoint['step']
        else:
            step = 0
        if model_ema is not None:
            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        print("With optim & sched!")

    return start_epoch-1, step

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    # model_ema._load_checkpoint(mem_file)
    model_ema.module.load_state_dict(checkpoint)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == 'inf':
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm