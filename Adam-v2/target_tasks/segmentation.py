"""
Sample run for heart segmentation

for i in {1..5}
do

python segmentation.py --dataset SCR-Heart  \
--arch convnext_original_upernet2  \
--init adamv2_convnext --batch_size 32 --lr 0.0002 \
--train_data_file /path/to/file \
--val_data_file /path/to/file \
--test_data_file /path/to/file \
--train_image_path /path/to/images \
--valid_image_path /path/to/images \
--test_image_path /path/to/images \
--train_mask_path /path/to/masks \
--valid_mask_path /path/to/masks \
--test_mask_path /path/to/masks \
--checkpoint /path/to/save_dir \
--encoder_weights /path/to/pretrained_model \
--model convnext_base  --run ${i} --optimizer adamw --normalization imagenet

done
"""
from convnext_segmentation import UperNet_convnext
import torch
import argparse
import os
from utils_segmentation import AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, step_decay,iou,mean_iou
import torch.backends.cudnn as cudnn
from torch import optim as optim
from timm.utils import NativeScaler, ModelEma
import numpy as np
import math
import sys
from dataset_segmentation import PNEDataset,SCRDataset,ChestXDetDataset,VinDrRibCXRDataset
import time
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform,
    RGBShift, Blur, GaussNoise,CenterCrop,GaussNoise,OpticalDistortion,RandomSizedCrop
)
import torch.nn.functional as F
from PIL import Image
import cv2


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=24,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
parser.add_argument('--patience', type=int, default=50, help='number of training patience')
parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")
parser.add_argument('--arch', dest='arch', default="swin_uneter", type=str)
parser.add_argument('--dataset', dest='dataset', default="pxs", type=str)
parser.add_argument('--run', dest='run', default="0", type=str)
parser.add_argument('--img_size', type=int, default=224, help='number of training patience')
parser.add_argument('--num_classes', dest='num_classes', default=1, type=int)
parser.add_argument('--test_only', dest='test_only', action="store_true")
parser.add_argument('--normalization', help='imagenet|None', default="chestx-ray")
parser.add_argument('--annotation_percent', help='annotation_percent', default=100, type=int)
parser.add_argument('--train_data_file', default=None)
parser.add_argument('--val_data_file', default=None)
parser.add_argument('--test_data_file', default=None)
parser.add_argument('--train_image_path', default=None)
parser.add_argument('--valid_image_path', default=None)
parser.add_argument('--test_image_path', default=None)
parser.add_argument('--train_mask_path', default=None)
parser.add_argument('--valid_mask_path', default=None)
parser.add_argument('--test_mask_path', default=None)
parser.add_argument('--organ', help='for scr dataset only: lung|heart|clavicle', default="heart")
parser.add_argument('--mode', help='train|test', default='train')
parser.add_argument('--checkpoint', help='model path', default="./Checkpoints")
parser.add_argument('--encoder_weights', help='imagenet|None', default=None)
parser.add_argument('--init', help='imagenet|None', default=None)
parser.add_argument('--model', help='model', default=None)
parser.add_argument("--optimizer", default='adamw', type=str)
parser.add_argument("--in_channels", dest="in_channels", help="in_channels", type=int, default=3)
parser.add_argument("--min_crop_size", dest="min_crop_size", help="min_crop_size", type=int, default=156)

args = parser.parse_args()

def build_model(conf):
    if conf.arch == "convnext_upernet2":
        model = UperNet_convnext(conf.model,img_size=conf.img_size, num_classes=conf.num_classes)
        if conf.encoder_weights is not None:
            print("Loading  pretrained weights", file=conf.log_writter)
            checkpoint = torch.load(conf.encoder_weights, map_location='cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            import re
            out_dict = {}
            for k, v in checkpoint_model.items():
                k = k.replace('stem.','downsample_layers.0.')
                k = re.sub(r'stages.([0-9]+).blocks.([0-9]+)', r'stages.\1.\2', k)
                k = re.sub(r'stages.([0-9]+).downsample.([0-9]+)', r'downsample_layers.\1.\2', k)
                k = k.replace('conv_dw','dwconv' )
                k = k.replace('mlp.fc','pwconv')
                if 'grn' in k:
                    k = k.replace('mlp.grn.bias','grn.beta' )
                    k = k.replace('mlp.grn.weight','grn.gamma')
                    v = v.reshape(v.shape[-1])
                k = k.replace('head.fc.','head.', )
                if v.ndim == 2 and 'head' not in k:
                    model_shape = model.backbone.state_dict()[k].shape
                    v = v.reshape(model_shape)
                out_dict[k] = v

            for k in list(out_dict.keys()):
                if k.startswith('head'):
                    del out_dict[k]
            msg = model.backbone.load_state_dict(out_dict, strict=False)
            print('Loaded with msg: {}'.format(msg))
    elif conf.arch == "convnext_original_upernet2":
        model = UperNet_convnext(conf.model,img_size=conf.img_size, num_classes=conf.num_classes)
        if conf.encoder_weights is not None:
            print("Loading  pretrained weights", file=conf.log_writter)
            checkpoint = torch.load(conf.encoder_weights, map_location='cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "teacher" in checkpoint:
                checkpoint = checkpoint["teacher"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "student" in checkpoint:
                checkpoint = checkpoint["student"]
            checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("encoder_q.", ""): v for k, v in checkpoint_model.items()}

            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            msg = model.backbone.load_state_dict(checkpoint_model, strict=False)
            print('Loaded with msg: {}'.format(msg))

    if conf.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, weight_decay=0, momentum=0.9, nesterov=False)
    model = model.double()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True

    loss_scaler = NativeScaler()
    return model, optimizer,loss_scaler

def train_one_epoch(model,train_loader, optimizer, loss_scaler, epoch,criterion = torch_dice_coef_loss ):
    model.train(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion = criterion
    end = time.time()

    for idx, (img,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = img.shape[0]


        img = img.double().cuda(non_blocking=True)
        mask = mask.double().cuda(non_blocking=True)
        outputs = torch.sigmoid(model(img))

        if outputs.size()[-1] != mask.size()[-1]:
            outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')

        loss = criterion(mask, outputs)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=args.log_writter)
            sys.exit(1)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses), file=args.log_writter)
            args.log_writter.flush()
            if args.debug_mode:
                break
    return losses.avg

def evaluation(model, val_loader, epoch,criterion = torch_dice_coef_loss):
    model.eval()
    losses = AverageMeter()
    criterion = criterion
    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            bsz = img.shape[0]
            img = img.double().cuda(non_blocking=True)
            mask = mask.double().cuda(non_blocking=True)
            outputs = torch.sigmoid(model(img))
            if outputs.size()[-1] != mask.size()[-1]:
                outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')
            loss = criterion(mask, outputs)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=args.log_writter)
                sys.exit(1)
            losses.update(loss.item(), bsz)
            torch.cuda.synchronize()
            if (idx + 1) % 10 == 0:
                print('Evaluation: [{0}][{1}/{2}]\t'
                      'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                    epoch, idx + 1, len(val_loader), ttloss=losses), file=args.log_writter)
                args.log_writter.flush()
                if args.debug_mode:
                    break
    return losses.avg

def test(test_loader, conf):
    if args.arch == "convnext_upernet2" or args.arch == "convnext_original_upernet2":
        model = UperNet_convnext(args.model,img_size=args.img_size, num_classes=args.num_classes)
    checkpoint = torch.load(os.path.join(args.model_path, 'ckpt.pth'), map_location='cpu')
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint_model)
    model = model.double()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True

    model.eval()
    with torch.no_grad():
        test_p = None
        test_y = None
        for idx, (img, mask) in enumerate(test_loader):
            with torch.cuda.amp.autocast():
                img = img.double().cuda(non_blocking=True)
                mask = mask.double().cuda(non_blocking=True)
                outputs = torch.sigmoid(model(img))
                if outputs.size()[-1] != mask.size()[-1]:
                    outputs = F.interpolate(outputs, size=args.img_size, mode='bilinear')
                outputs = outputs.cpu().detach()
                mask = mask.cpu().detach()
                if test_p is None and test_y is None:
                    test_p = outputs
                    test_y = mask
                else:
                    test_p = torch.cat((test_p, outputs), 0)
                    test_y = torch.cat((test_y, mask), 0)
                torch.cuda.empty_cache()
                if (idx + 1) % 20 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(test_loader)), file=args.log_writter)
                    args.log_writter.flush()
                    if args.debug_mode:
                        break
        print("Done testing iteration!", file=args.log_writter)
        args.log_writter.flush()
    test_p = test_p.numpy()
    test_y = test_y.numpy()
    test_y = test_y.reshape(test_p.shape)
    return test_y, test_p

def main(config):
    AUGMENTATIONS_TRAIN = Compose([
        OneOf([
            RandomBrightnessContrast(),
            RandomGamma(),
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
        RandomSizedCrop(min_max_height=(config.min_crop_size, config.img_size), height=config.img_size, width=config.img_size, p=0.25),
        ToFloat(max_value=1)
    ], p=1)

    if config.dataset == "PNE":
        train_dataset = PNEDataset(config.train_image_path, config.train_mask_path,
                                   transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                   annotation=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        validation_dataset = PNEDataset(config.valid_image_path, config.valid_mask_path, transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                        annotation=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        test_dataset = PNEDataset(config.test_image_path, config.test_mask_path,
                                  transforms=None, normalization=config.normalization,dim=(config.img_size,config.img_size,3))

    elif config.dataset == "SCR-Clavicle" or config.dataset == "SCR-Heart":
        train_dataset = SCRDataset(config.train_image_path, config.train_mask_path, config.train_data_file,
                                   transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                   anno_percent=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        validation_dataset = SCRDataset(config.valid_image_path, config.valid_mask_path, config.val_data_file,
                                        transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                        anno_percent=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        test_dataset = SCRDataset(config.test_image_path, config.test_mask_path, config.test_data_file, transforms=None,
                                  normalization=config.normalization,dim=(config.img_size,config.img_size,3))
    elif config.dataset == "ChestX-Det":
        train_dataset = ChestXDetDataset(config.train_image_path, config.train_mask_path, config.train_data_file,
                                         transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                         anno_percent=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        validation_dataset = ChestXDetDataset(config.valid_image_path, config.valid_mask_path, config.val_data_file,
                                              transforms=AUGMENTATIONS_TRAIN, normalization=config.normalization,
                                              anno_percent=config.annotation_percent,dim=(config.img_size,config.img_size,3))
        test_dataset = ChestXDetDataset(config.test_image_path, config.test_mask_path, config.test_data_file,
                                        transforms=None, normalization=config.normalization,dim=(config.img_size,config.img_size,3))

    elif config.dataset == "vindrribcxr":
         train_dataset = VinDrRibCXRDataset((config.train_image_path, config.train_mask_path),
                                            image_size=(config.img_size, config.img_size), mode="train",
                                            annotation=config.annotation_percent)
         validation_dataset = VinDrRibCXRDataset((config.valid_image_path, config.valid_mask_path),
                                                 image_size=(config.img_size, config.img_size), mode="val",
                                                 annotation=config.annotation_percent)
         test_dataset = VinDrRibCXRDataset((config.test_image_path, config.test_mask_path),
                                           image_size=(config.img_size, config.img_size), mode="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                       num_workers=config.num_workers,drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size,
                                                            shuffle=False, num_workers=config.num_workers,drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                      num_workers=config.num_workers,drop_last=False, pin_memory=True)

    if args.test_only:
        test_y, test_p = test(test_loader, config)
        print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=config.log_writter)
        print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=config.log_writter)
        print("[INFO] IoU  = {:.2f}%".format(100.0 * iou(test_y, test_p)), file=config.log_writter)
        print("[INFO] Mean IoU  = {:.2f}%".format(100.0 * mean_iou(test_y, test_p)), file=config.log_writter)
        config.log_writter.flush()
        exit(0)

    else:
        model, optimizer, loss_scaler = build_model(config)

    best_val_loss = 100000
    patience_counter = 0

    criterion = torch_dice_coef_loss
    for epoch in range(1, config.epochs):

        lr_ = step_decay(epoch, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'], epoch), file=config.log_writter)

        loss_avg = train_one_epoch(model,train_loader, optimizer, loss_scaler, epoch,criterion)
        print('Training loss: {}@Epoch: {}'.format(loss_avg, epoch), file=config.log_writter)
        config.log_writter.flush()


        val_avg = evaluation(model,val_loader,epoch,criterion)

        if val_avg < best_val_loss:
            save_file = os.path.join(config.model_path, 'ckpt.pth')
            save_model(model, optimizer, config, epoch + 1, save_file)


            print( "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss,val_avg, save_file), file=config.log_writter)
            best_val_loss = val_avg
            patience_counter = 0
        else:
            print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss), file=config.log_writter)
            patience_counter += 1
        if patience_counter > config.patience:
            print("Early Stopping", file=config.log_writter)
            break

        config.log_writter.flush()
        if config.debug_mode:
            break


    test_y, test_p = test(test_loader, config)

    print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=config.log_writter)
    print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=config.log_writter)
    print("[INFO] IoU  = {:.2f}%".format(100.0 * iou(test_y, test_p)), file=config.log_writter)
    print("[INFO] Mean IoU  = {:.2f}%".format(100.0 * mean_iou(test_y, test_p)), file=config.log_writter)
    config.log_writter.flush()

if __name__ == '__main__':
    args.debug_mode = False
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.encoder_weights is not None:
        args.model_path = os.path.join(args.checkpoint,args.dataset, args.arch, args.init,
                                  str(args.run))
    else:
        args.model_path = os.path.join(args.checkpoint, args.dataset,args.arch,str(args.model)+"_random", str(args.run))

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    logs_path = os.path.join(args.model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    args.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')
    if args.gpu is not None:
            args.device = "cuda"
    else:
            args.device = "cpu"
    print("devise:",args.device)
    main(args)