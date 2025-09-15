import os
import time
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from timm.models.layers import trunc_normal_
# ---------------------------------------Classification model------------------------------------
def Classifier_model(arch_name, num_class, conv=None, weight=None, linear_classifier=False, sobel=False,
                     activation=None,in_channels=3):
    if weight is None:
        weight = "none"
    if conv is None:
        if arch_name.lower().startswith("resnet"):
            model = models.__dict__[arch_name](pretrained=False)
        elif arch_name.lower().startswith("convnext"):
            print("creating Convnext model...")
            model = timm.create_model(arch_name, num_classes=num_class, pretrained=False)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if in_channels ==1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        if linear_classifier:
            for name, param in model.named_parameters():
                if name not in ['fc.0.weight', 'fc.0.bias', 'fc.weight', 'fc.bias']:
                    param.requires_grad = False

        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()

    def _weight_loading_check(_arch_name, _activation, _msg):
        if len(_msg.missing_keys) != 0:
            if _arch_name.lower().startswith("resnet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                else:
                    assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
            elif _arch_name.lower().startswith("densenet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
                else:
                    assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}

    state_dict = None
    if weight.lower() == "random"  or weight.lower() == "none":
        state_dict = model.state_dict()

    elif weight.lower() == "imagenet":
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()

        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded ImageNet pre-trained model")

    elif arch_name.lower().startswith("convnext") and "mimic" in weight.lower():
        state_dict = torch.load(weight, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}

        for k in list(state_dict.keys()):
            if k.startswith('head'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(weight))
        print("missing keys:", msg.missing_keys)

    elif arch_name.lower().startswith("convnext") and (not arch_name.lower().startswith("convnextv2")) and (weight is not None) and ("random" not in weight.lower()):
        import re
        print("=> loading checkpoint '{}'".format(weight))
        state_dict = torch.load(weight, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif "student" in state_dict:
            state_dict = state_dict["student"]


        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('downsample_layers.0.', 'stem.')
            k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
            k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
            k = k.replace('dwconv', 'conv_dw')
            k = k.replace('pwconv', 'mlp.fc')
            if 'grn' in k:
                k = k.replace('grn.beta', 'mlp.grn.bias')
                k = k.replace('grn.gamma', 'mlp.grn.weight')
                v = v.reshape(v.shape[-1])
            k = k.replace('head.', 'head.fc.')
            if k.startswith('norm.'):
                k = k.replace('norm', 'head.norm')
            if v.ndim == 2 and 'head' not in k:
                model_shape = model.state_dict()[k].shape
                v = v.reshape(model_shape)
            out_dict[k] = v

        for k in list(out_dict.keys()):
            if k.startswith('head'):
                del out_dict[k]
        msg = model.load_state_dict(out_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(weight))
        print(msg)

    elif  (arch_name.lower().startswith("resnet")) and ("chess" in weight.lower() or "lvmmed" in weight.lower() or "chestxray14" in weight.lower() or "chexpert" in weight.lower() or  "places365" in weight.lower() or "coco" in weight.lower() or "vrl" in weight.lower() or "pcrl" in weight.lower() or "dc" in weight.lower() or "best_checkpoint" in weight.lower() or "checkpoint" in weight.lower()  or "ssl_in_domain" in weight.lower() or "ssl_imagenet" in weight.lower() or "restoration" in weight.lower() or  "byol" in weight.lower() or "deepcluster-v2" in weight.lower() or \
            "infomin" in weight.lower() or "insdis" in weight.lower() or "moco-v1" in weight.lower() or \
            "moco-v2" in weight.lower() or "pirl" in weight.lower() or "pcl-v1" in weight.lower() or \
            "pcl-v2" in weight.lower() or "sela-v2" in weight.lower() or "simclr-v1" in weight.lower() or \
            "simclr-v2" in weight.lower() or "swav" in weight.lower() or "barlowtwins" in weight.lower() or "imagenet21k" in weight.lower() or "simsiam" in weight.lower() or  "obow" in weight.lower() or "simsiam" in weight.lower() or "dino" in weight.lower() or "clsa" in weight.lower()) and ("resnet50_ssl_transfer" not in weight.lower()):
        if os.path.isfile(weight):
            print("=> loading checkpoint '{}'".format(weight))
            state_dict = torch.load(weight, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "teacher" in state_dict:  # for dino
                state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("teacher.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}

            for k in list(state_dict.keys()):
                if k.startswith('fc'):
                    del state_dict[k]

            if "moco" in weight.lower():
                print("moco detected")
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                        del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg)
            _weight_loading_check(arch_name, activation, msg)
        else:
            print("=> no checkpoint found at '{}'".format(weight))

    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    return model, state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def computeAUROC(dataGT, dataPRED, classCount=14):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC


def save_checkpoint(state, is_best, filename='model'):
    # torch.save(state, filename + '_checkpoint.pth.tar')
    if is_best:
        torch.save( state,filename + '.pth.tar')

# ----------------------------------Whether Experiment Exist----------------------------------
def experiment_exist(log_file, exp_name):
    if not os.path.isfile(log_file):
        return False

    with open(log_file, 'r') as f:
        line = f.readline()
        while line:
            # print(line)
            # if line.replace('\n', '') == exp_name:
            if line.startswith(exp_name):
                return True
            line = f.readline()

    return False

# ---------------------------Callback function for OptionParser-------------------------------
def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        if arg[:2] == "--" and len(arg) > 2:
            break
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)



