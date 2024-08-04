#!/bin/bash

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8  main.py  --datasets chestxray14  --output_dir ./checkpoints/warmup_loc  --batch_size_per_gpu 64  --arch resnet50  --dist_url 'tcp://localhost:10007' --datasets_config ./datasets_config.yaml --mode L --epochs 200-200-100

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py  --datasets chestxray14  --output_dir ./checkpoints/adam-v2  --batch_size_per_gpu 64  --arch resnet50  --dist_url 'tcp://localhost:10007' --datasets_config ./datasets_config.yaml --mode LCD --epochs 10-90-165  --weights ./checkpoints/warmup_loc/checkpoint.pth

