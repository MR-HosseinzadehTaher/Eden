
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder
from data_loader import AnatomyDecomposerDataset
import data_loader
from utils import *
from engine import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Adam Training')
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',choices=model_names,help='model architecture')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='start epoch number')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N',help='mini-batch size, this is the total batch size of all GPUs on the current node when ')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum of SGD solver')
parser.add_argument('-p', '--print-freq', default=1, type=int,metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint')
parser.add_argument('--world-size', default=-1, type=int,help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,help='distributed backend')
parser.add_argument('--seed', default=None, type=int,help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',help='Use multi-processing distributed training')
parser.add_argument('--moco-dim', default=128, type=int,help='feature dimension')
parser.add_argument('--moco-k', default=65536, type=int,help='queue size')
parser.add_argument('--moco-m', default=0.999, type=float,help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.07, type=float,help='softmax temperature')
parser.add_argument('--mlp', action='store_true',help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',help='use cosine lr schedule')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='path to checkpoint directory')
parser.add_argument('--train_list', default=None, type=str,help='file for training list')
parser.add_argument('--val_list', default=None, type=str,help='file for validation list')
parser.add_argument('--encoder_weights', default=None, type=str,help='encoder pre-trained weights')
parser.add_argument('--weights', default=None, type=str,help='pre-trained weights')
parser.add_argument('--exp_name', default="none", type=str,help='experiment name')
parser.add_argument('--n', default=0, type=int,help=' data granularity level')

parser.add_argument('--crop_scale_min', default=0.2, type=float,help='min scale for random crop augmentation')
parser.add_argument('--sim_threshold', default=0.8, type=float,help='similarity threshold for purposive pruner')
parser.add_argument('--optimizer', default="sgd", type=str,help='adamw|sgd optimizer')


def main():
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir,args.exp_name)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # =======================create model=======================
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,encoder_weights=args.encoder_weights)

    if args.weights is not None:
        print("=> loading checkpoint '{}'".format(args.weights))
        state_dict = torch.load(args.weights, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("missing keys:", msg)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # ======================= define loss function and optimizer =======================
    if args.n == 0:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = PurposivePrunerLoss(args.sim_threshold).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_loss = 10000000000
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # ======================= create dataset and dataloader =======================
    dataset_train = AnatomyDecomposerDataset(pathImageDirectory=args.data, pathDatasetFile=args.train_list,
                                             augment=data_loader.Transform(crop_scale=(args.crop_scale_min, 1.)), n=args.n)
    dataset_valid = AnatomyDecomposerDataset(pathImageDirectory=args.data, pathDatasetFile=args.val_list,
                                             augment=data_loader.Transform(crop_scale=(args.crop_scale_min, 1.)), n=args.n)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=valid_sampler, drop_last=True)

    # ======================= training =======================
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        valid_loss = validate(valid_loader, model, criterion, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss':best_loss
            },  filename=os.path.join(args.checkpoint_dir,'checkpoint.pth'))

            print ("validation loss is: ",valid_loss)
            if valid_loss < best_loss:
                print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}".format(epoch, best_loss, valid_loss))
                best_loss = valid_loss
                torch.save(model.module.encoder_q.state_dict(),
                           os.path.join(args.checkpoint_dir, 'best_checkpoint.pth'))

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        torch.save(model.module.encoder_q.state_dict(), os.path.join(args.checkpoint_dir, 'last_checkpoint.pth'))

if __name__ == '__main__':
    main()
