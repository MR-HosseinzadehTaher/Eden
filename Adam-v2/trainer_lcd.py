
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
import models
from data_loader import CXRDatasetHierarchical
import shutil
import utils
from models import LocalizabilityHead
from utils import get_config
import convnext as convnext

def train_lcd(args):
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
    transform = Transform(args.global_crops_scale)

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
    args.embed_dim = embed_dim

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

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    compose_head = nn.Sequential(nn.Linear(args.n_parts * embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    decompose_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, args.n_parts * embed_dim))

    model = models.Adamv2(student, teacher, compose_head, decompose_head, args.local_crops_number + 1, args.n_parts)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # ============ preparing loss ... ============
    localizability_loss = LocalizabilityLoss(
        args.out_dim,
        args.local_crops_number + 1,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.total_epochs,
    ).cuda()

    composability_loss= torch.nn.MSELoss().cuda()
    decomposability_loss= torch.nn.MSELoss().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
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
    start_epoch = 0
    to_restore = {"epoch": 0}
    if args.weights is not None:
        utils.restart_from_checkpoint(
            args.weights,
            run_variables=to_restore,
            student=model.module.student,
            teacher=model.module.teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            localizability_loss=localizability_loss,
        )
    print("Starting training !")
    start_time = time.time()

    for step in range(len(epochs)):
        concat_datasets = []
        for dataset_name in args.datasets:
            dataset = CXRDatasetHierarchical(images_path=datasets_config[dataset_name]['data_dir'], file_path=datasets_config[dataset_name]['train_list'],
                                             augment=transform, data_granularity=data_granularity[step], nb_parts=args.n_parts)

            concat_datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(concat_datasets)
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if step == 0:
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
        for epoch in range(start_epoch, start_epoch + epochs[step]):
            train_data_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(model, localizability_loss, composability_loss, decomposability_loss,
                                          train_data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                          epoch, fp16_scaler, args)
            save_dict = {
                'model': model.state_dict(),
                'student': model.module.student.state_dict(),
                'teacher': model.module.teacher.state_dict(),
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
        start_epoch = start_epoch + epochs[step]
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(model, localizability_criterion, composability_criterion, decomposability_criterion, data_loader,
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
            student_locality,teacher_locality,student_compose,teacher_compose,student_decompose, teacher_decompose = model(images)

            localizability_loss = localizability_criterion(student_locality, teacher_locality, epoch)
            composability_loss = composability_criterion(student_compose, teacher_compose)
            decomposability_loss = 0
            for i in range(len(teacher_decompose)):
                decomposability_loss += decomposability_criterion(student_decompose[:, i * args.embed_dim:(i + 1) * args.embed_dim],
                                                            teacher_decompose[i])
            decomposability_loss = decomposability_loss / len(teacher_decompose)
            loss = args.localizability_weight * localizability_loss + args.composability_weight * composability_loss + args.decomposability_weight * decomposability_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model.module.student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model.module.student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(model.module.student.parameters(), model.module.teacher.parameters()):
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
        teacher_out = teacher_out.detach().chunk(1)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
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
    def __init__(self, global_crops_scale):
        self.global_crops_scale=global_crops_scale
        basic_trans = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomRotation(10)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transform1 = transforms.Compose([
            transforms.Resize((224,224), interpolation=Image.BICUBIC),
            basic_trans,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        self.global_transform2 = transforms.Compose([
            basic_trans,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.extract_local_crops =transforms.Compose([RandomCenterCrop(96, interpolation=Image.BICUBIC)])
        self.extract_global_crops =transforms.Compose([RandomCenterCrop(224, interpolation=Image.BICUBIC)])

        self.local_transform = transforms.Compose([
            basic_trans,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, x):
        crops = []
        crops.append(self.global_transform1(x[0]))
        crops.append(self.global_transform2(self.extract_global_crops((x[0], self.global_crops_scale, (3.0 / 4.0, 4.0 / 3.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.05, 0.1), (3.0 / 4.5, 1.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.1, 0.2), (1.0, 4.5 / 3.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.2, 0.3), (3.0 / 4.5, 1.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.3, 0.4), (1.0, 4.5 / 3.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.4, 0.5), (3.0 / 4.5, 1.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.5, 0.6), (1.0, 4.5 / 3.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.6, 0.7), (3.0 / 4.5, 1.0)))))
        crops.append(self.local_transform(self.extract_local_crops((x[0], (0.7, 0.8), (1.0, 4.5 / 3.0)))))
        for cr in x[1:]:
            crops.append(self.global_transform1(cr))
        return crops

class RandomCenterCrop(object):
    def __init__(
            self,
            size,
            interpolation,

    ):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        width,height = img.size
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = (height - h) // 2
                j = (width - w) // 2
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    def __call__(self, data ):
        img, scale,ratio = data
        i, j, h, w = self.get_params(img, scale, ratio)
        output = transforms.functional.crop(img, i, j, h, w)
        output = transforms.functional.resize(output, (self.size,self.size),self.interpolation)
        return output

