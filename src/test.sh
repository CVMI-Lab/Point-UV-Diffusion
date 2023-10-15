#!/bin/bash

# test the coarse stage of image-condition model on the table dataset
python -m torch.distributed.launch --master_port 1234 --nproc_per_node 2 test.py \
experiment=table_image_coarse \
ckpt_name=../pretrain/image_condition/table_image_coarse/ckpt/ckpt.pth

# test the fine stage of image-condition model on the table dataset
python -m torch.distributed.launch --master_port 1234 --nproc_per_node 2 test.py \
experiment=table_image_fine \
ckpt_name=../pretrain/image_condition/table_image_fine/ckpt/ckpt.pth \
datamodule.data_detail.finetune_folder=../results/test/pretrain/image_condition/table_image_coarse/static_timestamped \
datamodule.data_detail.use_stage1_test=True