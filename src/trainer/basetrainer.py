import math
import time
import sys
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
import pdb
from src.utils import utils
from pytorch_lightning.utilities import rank_zero_only
from torch.cuda.amp import autocast as autocast

class MyBaseTrainer:
    def __init__(
        self,
        min_epochs,
        max_epochs,
        distributed=True,
        logger=None,
        device='cuda',
        step=0,
        ckpt_resume=False,
        modelmodule=None,
    ):
        super().__init__()

        self.distributed = distributed
        self.device = device
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.logger = logger
        self.step = step
        self.ckpt_resume = ckpt_resume
        if self.ckpt_resume:
            self.min_epochs = self.load_model(modelmodule)

    def train_one_epoch(self, data_loader, modelmodule):
        for batch in data_loader:
            # if step >= num_training_steps_per_epoch:
            #    continue
            torch.cuda.synchronize()
            time_st = time.time()
            modelmodule.net.zero_grad()
            modelmodule.optimizer.zero_grad()

            batch_stat, batch_media = modelmodule.training_step(batch, self.step)

            loss = batch_stat['loss']
            image = batch_media['image']
            self.logger.save_image(image, self.step)

            if 'render_image' in batch_media.keys():
                render_image = batch_media['render_image']
                self.logger.save_image(render_image, self.step, name='render')

            loss.backward()
            modelmodule.optimizer.step()
            modelmodule.optimizer.zero_grad()
            if modelmodule.net_ema is not None:
                modelmodule.net_ema.update(modelmodule.net)

            self.step += 1

            torch.cuda.synchronize()
            time_iter = time.time()-time_st
            batch_stat['time_iter'] = time_iter
            self.logger.print_stat(batch_stat, self.step)



    def train(self, datamodule, modelmodule, ckpt_path=None):
        start_epoch = self.min_epochs
        end_epoch = self.max_epochs

        modelmodule.net.train()
        data_loader_train = datamodule.train_dataloader()
        data_loader_val = datamodule.val_dataloader()
        # start training!!!
        print(f"Start training for {self.max_epochs} epochs")
        for epoch in range(start_epoch, end_epoch):
            if self.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            self.train_one_epoch(data_loader_train, modelmodule)
            self.save_model(epoch, modelmodule)

    def load_model(self, modelmodule):
        net = modelmodule.net
        net_without_ddp = modelmodule.net_without_ddp
        net_ema = modelmodule.net_ema
        optimizer = modelmodule.optimizer
        output_dir = self.ckpt_resume
        epoch, step = utils.auto_load_model(
            output_dir=output_dir, model=net, model_without_ddp=net_without_ddp, optimizer=optimizer,
            model_ema=net_ema)
        print('load epoch %s and step %s' % (epoch, step))
        self.step = step
        return epoch



    @rank_zero_only
    def save_model(self, epoch, modelmodule):
        net = modelmodule.net
        net_without_ddp = modelmodule.net_without_ddp
        net_ema = modelmodule.net_ema
        optimizer = modelmodule.optimizer
        output_dir = self.logger.save_ckpt_dir

        print('save ckpt*************************')
        if (epoch + 1) % self.logger.save_ckpt_freq == 0 or epoch + 1 == self.max_epochs:
            utils.save_model(
                output_dir=output_dir, model=net, model_without_ddp=net_without_ddp, optimizer=optimizer,
                epoch=epoch, step=self.step, model_ema=net_ema)
        utils.save_model(
            output_dir=output_dir, model=net, model_without_ddp=net_without_ddp, optimizer=optimizer,
            epoch=epoch, step=self.step, model_ema=net_ema, epoch_name='latest')