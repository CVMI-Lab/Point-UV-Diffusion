import os
from typing import Optional
import torch
from pytorch_lightning.utilities import rank_zero_only
import torchvision

class MyBaseLogger:
    def __init__(self, log_dir, log_freq, log_img_freq, save_ckpt_freq):
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.log_img_freq = log_img_freq
        self.save_ckpt_freq = save_ckpt_freq
        self.set_up()

    @rank_zero_only
    def set_up(self):
        os.makedirs(self.log_dir, exist_ok=True)
        filename = os.path.join(self.log_dir, 'mylog.log')
        self.file = open(filename, "wt")

        self.image_dir = os.path.join(self.log_dir, 'image')
        os.makedirs(self.image_dir, exist_ok=True)

        self.save_ckpt_dir = os.path.join(self.log_dir, 'ckpt')
        os.makedirs(self.save_ckpt_dir, exist_ok=True)

    @rank_zero_only
    def save_command(self, command_line):
        filename = os.path.join(self.log_dir, 'command.txt')
        with open(filename, "wt") as file:
            file.write(command_line)

    @rank_zero_only
    def writekvs(self, kvs):
        keywidth, valwidth = 10, 30
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in sorted(kvs.items(), key=lambda kv: kv[0].lower()):
            valstr = "%-8.3g" % val if hasattr(val, "__float__") else str(val)
            line_str = "| {key:<{kw}} | {val:<{vw}} |".format(key=self._truncate(key), kw=keywidth, val=self._truncate(valstr), vw=valwidth)
            lines.append(line_str)
            print(line_str)
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def _truncate(self, s):
        return s[:27] + "..." if len(s) > 30 else s

    @rank_zero_only
    def print_stat(self, stat, step):
        if step % self.log_freq == 0:
            stat['step'] = step
            self.writekvs(stat)

    @rank_zero_only
    def save_image(self, image_batch, step, name=''):
        if step % self.log_img_freq == 0 and image_batch is not None:
            filename = os.path.join(self.image_dir, f'{name}{step:08d}.jpg')
            torchvision.utils.save_image(image_batch.detach().cpu(), filename)
