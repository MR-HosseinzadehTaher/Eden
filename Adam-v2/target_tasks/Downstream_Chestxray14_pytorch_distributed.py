# running command
"""
for i in {1..5}
do

OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --rdzv_backend=c10d \
--rdzv_endpoint=localhost:29406  Downstream_Chestxray14_pytorch_distributed.py \
--model convnext_base --init adamv2_convnext_base \
--mode train  --normalization imagenet  --augment full  --batch_size 32 \
--optimizer Adam --lr 1e-4  --lr_Scheduler ReduceLROnPlateau  \
--proxy_dir /path/to/pretrained_model  --dist_url tcp://localhost:10006  --run ${i} --img_size 448 --workers 5
--data_dir /path/to/dataset \
--train_list /path/to/train_file \
--val_list /path/to/validation_file \
--test_list /path/to/test_file \
done

"""
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm
import torch.distributed as dist
from model_pytorch import experiment_exist, vararg_callback_bool, vararg_callback_int
from dataloader_pytorch import Augmentation, ChestX_ray14
from model_pytorch import Classifier_model, AverageMeter, ProgressMeter, computeAUROC, \
  save_checkpoint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import misc as misc
sys.setrecursionlimit(40000)
from torch.utils.tensorboard import SummaryWriter

parser = OptionParser()

parser.add_option("--data_dir", dest="data_dir",
                  help="path to images",
                  default="Data/ChestX-ray14/images", type="string")
parser.add_option("--train_list", dest="train_list", help="file for training list",
                  default="dataset/Xray14_train_official.txt", type="string")
parser.add_option("--val_list", dest="val_list", help="file for validating list",
                  default="dataset/Xray14_val_official.txt", type="string")
parser.add_option("--test_list", dest="test_list", help="file for test list",
                  default="dataset/Xray14_test_official.txt", type="string")
# network architecture
parser.add_option("--model", dest="model_name", help="convnext_base", default="convnext_base", type="string")
parser.add_option("--init", dest="init",
                  help="initialization method",
                  default="Random", type="string")
parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                  default=14, type="int")
# data loader
parser.add_option("--data_set", dest="data_set", default="ChesstXray14", type="string")
parser.add_option("--normalization", dest="normalization", help="how to normalize data", default="imagenet",
                  type="string")
parser.add_option("--augment", dest="augment", help="full", default="full", type="string")
parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
# training detalis
parser.add_option("--linear_classifier", dest="linear_classifier", help="whether train a linear classifier",
                  default=False, action="callback", callback=vararg_callback_bool)
parser.add_option("--sobel", dest="sobel", help="Sobel filtering", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--mode", dest="mode", help="train | test | valid", default="train", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=1000, type="int")
parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
parser.add_option("--lr", dest="lr", help="learning rate", default=1e-4, type="float")
parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default=None, type="string")
parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                  callback=vararg_callback_bool)
parser.add_option('--run', dest="run",default=0, type="int")
parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=1, type="int")
parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                  default=True, action="callback", callback=vararg_callback_bool)
parser.add_option("--activate", dest="activate", help="activation", default="Sigmoid", type="string")
# pretrained weights
parser.add_option("--proxy_dir", dest="proxy_dir", help="Pretrained model folder", default=None, type="string")
parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")

# distributed training parameters
parser.add_option('--world_size', dest="world_size", default=1, type="int",
                    help='number of distributed processes')
parser.add_option('--local_rank', dest="local_rank",default=-1, type="int")
parser.add_option('--dist_on_itp', dest="dist_on_itp", default=False, action="callback", callback=vararg_callback_bool)
parser.add_option('--dist_url', dest="dist_url",default='env://', help='url used to set up distributed training')
parser.add_option('--model_path', dest="model_path",default='./Models/ChestXray14/', help='path to save checkpoints')

(options, args) = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
misc.init_distributed_mode(options)
cudnn.benchmark = True

model_path = options.model_path

options.exp_name = options.model_name + "_" + options.init
output_path = "./Outputs/ChestXray14/"
options.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                       'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                       'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
if misc.get_rank() == 0:
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(os.path.join(model_path, options.exp_name)):
        os.makedirs(os.path.join(model_path, options.exp_name))
dist.barrier()

if options.normalization == "default":
    if options.init.lower() == "random":
        options.normalization = "chestx-ray"
    elif options.init.lower() == "imagenet":
        options.normalization = "imagenet"


def train(train_loader, model, criterion, optimizer, epoch,loss_scaler):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  progress = ProgressMeter(
    len(train_loader),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))
  model.train()
  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    varInput, varTarget = input.float().to(device, non_blocking=True), target.float().to(device, non_blocking=True)
    with torch.cuda.amp.autocast():
      varOutput = model(varInput)
      loss = criterion(varOutput, varTarget)

    optimizer.zero_grad()
    loss_scaler.scale(loss).backward()
    loss_scaler.unscale_(optimizer)
    loss_scaler.step(optimizer)
    loss_scaler.update()

    torch.cuda.synchronize()
    losses.update(loss.item(), varInput.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % options.print_freq == 0:
      progress.display(i)

def validate(val_loader, model, criterion):
  model.eval()
  with torch.no_grad():
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
      len(val_loader),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, (input, target) in enumerate(valid_data):
      varInput, varTarget = input.float().to(device, non_blocking=True), target.float().to(device, non_blocking=True)

      varOutput = model(varInput)

      loss = criterion(varOutput, varTarget)

      torch.cuda.synchronize()
      losses.update(loss.item(), varInput.size(0))
      losses.update(loss.item(), varInput.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % options.print_freq == 0:
        progress.display(i)

  return losses.avg

def test(model, test_loader, config):
  model.eval()
  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (input, target) in enumerate(tqdm(test_loader)):
      target = target.cuda()
      y_test = torch.cat((y_test, target), 0)

      if len(input.size()) == 4:
        bs, c, h, w = input.size()
        n_crops = 1
      elif len(input.size()) == 5:
        bs, n_crops, c, h, w = input.size()

      varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

      out = model(varInput)

      out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

augment = Augmentation(normalize=options.normalization).get_augmentation(
"{}_{}".format(options.augment, options.img_size), "train")

datasetTrain = ChestX_ray14(pathImageDirectory=options.data_dir, pathDatasetFile=options.train_list,
                            augment=augment, anno_percent=options.anno_percent)

augment = Augmentation(normalize=options.normalization).get_augmentation(
"{}_{}".format(options.augment, options.img_size), "valid")
datasetVal = ChestX_ray14(pathImageDirectory=options.data_dir, pathDatasetFile=options.val_list,
                          augment=augment, anno_percent=options.anno_percent)


if True:  # args.distributed:
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
      datasetTrain, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.DistributedSampler(
      datasetVal, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    train_data = torch.utils.data.DataLoader(
        datasetTrain, sampler=sampler_train,
        batch_size=options.batch_size,
        num_workers=options.workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_data = torch.utils.data.DataLoader(
        datasetVal, sampler=sampler_val,
        batch_size=options.batch_size,
        num_workers=options.workers,
        pin_memory=True,
        drop_last=False
    )

num_tasks = misc.get_world_size()
global_rank = misc.get_rank()

augment = Augmentation(normalize=options.normalization).get_augmentation(
"{}_{}".format(options.augment, options.img_size), "test", options.test_augment)

datasetTest = ChestX_ray14(pathImageDirectory=options.data_dir, pathDatasetFile=options.test_list,
                                 augment=augment)

sampler_test = torch.utils.data.SequentialSampler(datasetTest)

test_data = torch.utils.data.DataLoader(
    datasetTest, sampler=sampler_test,
    batch_size=options.batch_size, #this is batch size per gpu
    num_workers=options.workers,
    pin_memory=True,
    drop_last=False,
)
print(" dataloaders created")

exp_name = options.exp_name + "_run_" + str(options.run)
init_epoch = 0
init_loss = 100000

if options.init.lower() == "imagenet" or options.init.lower() == "random":
    model, _ = Classifier_model(options.model_name.lower(), options.num_class, weight=options.init,
                                linear_classifier=options.linear_classifier, activation=None)

elif options.proxy_dir is not None:
    model, _ = Classifier_model(options.model_name.lower(), options.num_class, weight=options.proxy_dir,
                                linear_classifier=options.linear_classifier, sobel=options.sobel,
                                activation=None)
print(model)
model.to(device)
model_without_ddp = model

if options.distributed:
    print("distributed training")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[options.gpu])
    model_without_ddp = model.module

parameters = list(filter(lambda p: p.requires_grad, model_without_ddp.parameters()))

if options.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(parameters, lr=options.lr)
elif options.optimizer.lower() == "adamw":
    optimizer = torch.optim.AdamW(parameters, lr=options.lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=options.patience // 2, mode='min',
                              threshold=0.0001, min_lr=0, verbose=True)

loss_scaler = torch.cuda.amp.GradScaler()

loss = torch.nn.BCEWithLogitsLoss().cuda()

lossMIN = init_loss
patience_cnt = 0
_model_path = os.path.join(model_path, options.exp_name, exp_name + '_checkpoint.pth.tar')

# training phase
if options.mode == "train":
  # training phase
    print("start training")

    for epochID in range(init_epoch, options.num_epoch):
      if options.distributed:
            train_data.sampler.set_epoch(epochID)

      train(train_data, model, loss, optimizer, epochID,loss_scaler)

      val_loss = validate(valid_data, model, loss)

      scheduler.step(val_loss)

      if val_loss < lossMIN:
        print(
          "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epochID, lossMIN, val_loss,
                                                                                             _model_path))
        lossMIN = val_loss
        patience_cnt = 0
        save_checkpoint={
          'epoch': epochID + 1,
          'lossMIN': lossMIN,
          'state_dict': model_without_ddp.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'scaler': loss_scaler.state_dict(),
        }
        misc.save_on_master(save_checkpoint, _model_path)

      else:
        patience_cnt += 1

      if patience_cnt > options.patience:
        break
print("start testing")

pathCheckpoint = os.path.join(model_path, options.exp_name,exp_name+'_checkpoint.pth.tar')
modelCheckpoint = torch.load(pathCheckpoint)
state_dict = modelCheckpoint['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('module.'):
        # remove prefix
        state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
msg = model_without_ddp.load_state_dict(state_dict)
assert len(msg.missing_keys) == 0
print("=> loaded pre-trained model '{}'".format(pathCheckpoint))

output_file = os.path.join(output_path, options.exp_name + "_test.txt")
y_test, p_test = test(model, test_data, options)

if misc.get_rank() == 0:
    aurocIndividual = computeAUROC(y_test, p_test, options.num_class)
    print(">>{}: AUC = {}".format(exp_name, options.class_name))
    print(">>{}: AUC = {}".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator=',')))
    with open(output_file, 'a') as f_w:
        f_w.write("{}: AUC = {}\n".format(exp_name, options.class_name))
        f_w.write(
            "{}: AUC = {}\n".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator='\t')))
        if 'No_Finding' in options.class_name:
            index = aurocIndividual.index('No_Finding')
            aurocIndividual.pop(index)
        aurocMean = np.array(aurocIndividual).mean()
        print(">>{}: AUC = {:.4f}".format(exp_name, aurocMean))
        f_w.write("{}: ACC = {:.4f}\n".format(exp_name, aurocMean))
print("Done!")



