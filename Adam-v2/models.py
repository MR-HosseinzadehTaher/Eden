import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import yaml
from utils import trunc_normal_

class LocalizabilityHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x,apply_head=True):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            output = torch.cat((output, _out))
            start_idx = end_idx
        if apply_head:
            return self.head(output)
        else:
            return output

class Adamv2(nn.Module):
    def __init__(self, student, teacher, compose_head, decompose_head,nb_crops,nb_parts):
        super(Adamv2, self).__init__()
        self.student = student
        self.teacher = teacher
        self.compose_head = compose_head
        self.decompose_head = decompose_head
        self.nb_crops=nb_crops
        self.nb_parts=nb_parts

    def forward(self, x):
        # Locality
        teacher_locality = self.teacher(x[0],apply_head=True)
        student_locality = self.student(x[1:self.nb_crops+1],apply_head=True)
        # Composability
        student_parts = torch.FloatTensor().cuda()
        for i in range(self.nb_crops+1,self.nb_crops+1+self.nb_parts):
            p_s = self.student(x[i], apply_head=False)
            student_parts = torch.cat((student_parts, p_s), 1)
        student_compose= self.compose_head(student_parts)
        student_compose = nn.functional.normalize(student_compose, dim=1)
        teacher_compose = self.teacher(x[0],apply_head=False)
        teacher_compose = nn.functional.normalize(teacher_compose, dim=1)
        #Decomposability
        s_whole = self.student(x[0], apply_head=False)
        student_decompose = self.decompose_head(s_whole)
        student_decompose = nn.functional.normalize(student_decompose, dim=1)
        teacher_decompose=[]
        for i in range(self.nb_crops + 1, self.nb_crops + 1 + self.nb_parts):
            p_t = self.teacher(x[i], apply_head=False)
            p_t = nn.functional.normalize(p_t, dim=1)
            teacher_decompose.append(p_t.detach())
        return student_locality,teacher_locality,student_compose,teacher_compose.detach(),student_decompose,teacher_decompose