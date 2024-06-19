
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from data_loader import CXRDataset
import shutil
import utils
from models import LocalizabilityHead
from utils import get_config
import models
import convnext as convnext

def train_l(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing training params ... ============
    epochs = list(map(int, args.epochs.split('-')))
    data_granularity = list(map(int, args.data_granularity.split('-')))
    assert len(epochs) == len(data_granularity)

    datasets_config = get_config(args.datasets_config)
    args.total_epochs=sum(epochs)

    # ============ preparing augmentation ... ============
    transform = Transform(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number
    )

    # ============ building student and teacher networks ... ============
    if args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch in convnext.__dict__.keys():
        if args.arch =="convnext_base":
            embed_dim = 1024
        else:
            embed_dim = 768
        student = convnext.__dict__[args.arch]()
        teacher = convnext.__dict__[args.arch]()

    else:
        print(f"Unknow architecture: {args.arch}")

    if args.weights is not None:
            state_dict = torch.load(args.weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('head'):
                    del state_dict[k]

            msg = student.load_state_dict(state_dict, strict=False)
            print("missing keys:", msg)

    student = models.MultiCropWrapper(student, LocalizabilityHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = models.MultiCropWrapper(
        teacher,
        LocalizabilityHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    localizability_loss = LocalizabilityLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.total_epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ training ... ============
    start_time = time.time()
    print("Starting training....")
    start_epoch=0
    for step in range(len(epochs)):
        concat_datasets = []
        for dataset_name in args.datasets:
            dataset = CXRDataset(images_path=datasets_config[dataset_name]['data_dir'], file_path=datasets_config[dataset_name]['train_list'],
                                     augment=transform, data_granularity=data_granularity[step])

            concat_datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(concat_datasets)
        print(f"Data loaded: there are {len(train_dataset)} training images.")
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if step==0:
            # ============ init schedulers ... ============
            lr_schedule = utils.cosine_scheduler(
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
                args.min_lr,
                args.total_epochs, len(train_data_loader),
                warmup_epochs=args.warmup_epochs,
            )
            wd_schedule = utils.cosine_scheduler(
                args.weight_decay,
                args.weight_decay_end,
                args.total_epochs, len(train_data_loader),
            )
            momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                                       args.total_epochs, len(train_data_loader))
        for epoch in range(start_epoch, start_epoch+epochs[step]):
            train_data_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, localizability_loss,
                train_data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'localizability_loss': localizability_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        start_epoch=start_epoch+epochs[step]
        print("start_epoch",start_epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, localizability_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.total_epochs)
    for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = localizability_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LocalizabilityLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class Transform(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        basic_trans = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomRotation(30)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            basic_trans,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            basic_trans,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            basic_trans,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image,base_crop_size):
        crops = []
        x1 = transforms.Compose([transforms.RandomCrop(base_crop_size)])(image)
        crops.append(self.global_transform1(x1))
        crops.append(self.global_transform2(x1))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(x1))
        return crops

