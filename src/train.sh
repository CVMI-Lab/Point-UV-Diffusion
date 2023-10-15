#!/bin/bash

# train the coarse stage of image-condition model on the table dataset
python -m torch.distributed.launch --master_port 1234 --nproc_per_node 4 \
train.py experiment=table_image_coarse \
datamodule.batch_size=2