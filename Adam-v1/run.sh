#!/bin/bash

python -u main.py  /path/to/training/images --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos  --exp_name n0  --epochs 200 --workers 16  --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --checkpoint-dir ./checkpoints --n 0 --sim_threshold 0.8

python -u main.py  /path/to/training/images --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos  --exp_name n2  --epochs 200 --workers 16  --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --checkpoint-dir ./checkpoints  --weights ./checkpoints/n0/checkpoint.pth  --n 2 --sim_threshold 0.8


python -u main.py  /path/to/training/images --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos  --exp_name n4  --epochs 200 --workers 16  --train_list dataset/Xray14_train_official.txt --val_list dataset/Xray14_val_official.txt --checkpoint-dir ./checkpoints  --weights ./checkpoints/n2/checkpoint.pth  --n 4 --sim_threshold 0.8 
