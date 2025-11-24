'''
python Downstream_VinDrCXR_pytorch.py  --GPU 0  \
--init adam_v2  --mode train  --normalization imagenet \
--augment full  --batch_size 32  --optimizer Adam --lr 2e-4  --lr_Scheduler ReduceLROnPlateau  \
--trial 5 --start_index 0 --model  convnext_base \
--proxy_dir /path/to/pretrained_model  \
--data_dir /path/to/dataset  \
--train_list /path/to/train/split/file  \
--val_list /path/to/val/split/file  \
--test_list /path/to/test/split/file  \
'''

import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from model_pytorch import experiment_exist, vararg_callback_bool, vararg_callback_int
from dataloader_pytorch import Augmentation, VinDrCXR
from model_pytorch import Classifier_model, AverageMeter, ProgressMeter, computeAUROC, \
  save_checkpoint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                  callback=vararg_callback_int)
parser.add_option("--model", dest="model_name", help="convnext_base", default="convnext_base", type="string")
parser.add_option("--init", dest="init",
                  help="model initialization",
                  default="Adam_v2", type="string")
parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                  default=6, type="int")
parser.add_option("--data_set", dest="data_set", help="VinDR_CXR", default="VinDR_CXR", type="string")
parser.add_option("--normalization", dest="normalization", help="how to normalize data", default="imagenet",
                  type="string")
parser.add_option("--augment", dest="augment", help="full", default="full", type="string")
parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
parser.add_option("--data_dir", dest="data_dir",
                  help="path to images",
                  default="VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/", type="string")
parser.add_option("--train_list", dest="train_list", help="file for training list",
                  default=None, type="string")
parser.add_option("--val_list", dest="val_list", help="file for validating list",
                  default=None, type="string")
parser.add_option("--test_list", dest="test_list", help="file for test list",
                  default=None, type="string")
parser.add_option("--mode", dest="mode", help="train | test | valid", default="train", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=1000, type="int")
parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
parser.add_option("--lr", dest="lr", help="learning rate", default=2e-4, type="float")
parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default="ReduceLROnPlateau", type="string")
parser.add_option("--patience", dest="patience", help="num of patient epoches", default=30, type="int")
parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                  default=True, action="callback", callback=vararg_callback_bool)
parser.add_option("--proxy_dir", dest="proxy_dir", help="Pretrained model folder", default=None, type="string")
parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
parser.add_option("--in_channels", dest="in_channels", help="in_channels", default=3, type="int")
parser.add_option('--model_path', dest="model_path",default='./Models/VinDR_CXR/', help='path to save checkpoints')
parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")

(options, args) = parser.parse_args()
if options.GPU is not None:
  os.environ["CUDA_VISIBLE_DEVICES"] = str(options.GPU)[1:-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path =options.model_path
options.exp_name = options.model_name + "_" + options.init
model_path = os.path.join(model_path, options.exp_name)
if not os.path.exists(model_path):
  os.makedirs(model_path)
output_path = "./Outputs/VinDR_CXR/"
if not os.path.exists(output_path):
  os.makedirs(output_path)

options.class_name = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']

def train(train_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  progress = ProgressMeter(
    len(train_loader),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))
  model.train()
  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    varInput, varTarget = input.float().to(device), target.float().to(device)
    varOutput = model(varInput)
    loss = criterion(varOutput, varTarget)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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
    for i, (input, target) in enumerate(val_loader):
      varInput, varTarget = input.float().to(device), target.float().to(device)
      varOutput = model(varInput)
      loss = criterion(varOutput, varTarget)
      losses.update(loss.item(), varInput.size(0))
      batch_time.update(time.time() - end)
      end = time.time()
      if i % options.print_freq == 0:
        progress.display(i)
  return losses.avg

def test(pathCheckpoint, test_loader, config):
  model, _ = Classifier_model(config.model_name.lower(), config.num_class,activation=config.activate,in_channels=config.in_channels)
  print(model)
  modelCheckpoint = torch.load(pathCheckpoint)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]
  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(pathCheckpoint))
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)
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
      if "convnext" in config.model_name.lower():
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)
  return y_test, p_test

if options.mode == "train":
  log_file = os.path.join(model_path, "models.log")
  augment = Augmentation(normalize=options.normalization).get_augmentation(
    "{}_{}".format(options.augment, options.img_size), "train")
  datasetTrain = VinDrCXR(images_path=options.data_dir, file_path=options.train_list,
                              augment=augment,annotation_percent=options.anno_percent,in_channels=options.in_channels)
  augment = Augmentation(normalize=options.normalization).get_augmentation(
    "{}_{}".format(options.augment, options.img_size), "valid")
  datasetVal = VinDrCXR(images_path=options.data_dir, file_path=options.val_list,
                            augment=augment,annotation_percent=options.anno_percent,in_channels=options.in_channels)
  train_data = DataLoader(dataset=datasetTrain, batch_size=options.batch_size, shuffle=True,
                          num_workers=options.workers, pin_memory=True)
  valid_data = DataLoader(dataset=datasetVal, batch_size=options.batch_size, shuffle=False,
                          num_workers=options.workers, pin_memory=True)
  print("start training")
  for i in range(options.start_index, options.num_trial):
    experiment = options.exp_name + "_run_" + str(i)
    init_epoch = 0
    init_loss = 100000
    model, _ = Classifier_model(options.model_name.lower(), options.num_class, weight=options.proxy_dir,
                                  activation=options.activate)
    print(model)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if options.optimizer.lower() =="adam":
      optimizer = torch.optim.Adam(parameters, lr=options.lr)
    elif options.optimizer.lower() =="adamw":
      optimizer = torch.optim.AdamW(parameters, lr=options.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=options.patience // 2, mode='min',
                                  threshold=0.0001, min_lr=0, verbose=True)
    if "convnext" in options.model_name.lower():
      loss = torch.nn.BCEWithLogitsLoss()
    else:
      loss = torch.nn.BCELoss()

    lossMIN = init_loss
    patience_cnt = 0
    save_model_path = os.path.join(model_path, experiment)
    for epochID in range(init_epoch, options.num_epoch):
      train(train_data, model, loss, optimizer, epochID)
      val_loss = validate(valid_data, model, loss)
      scheduler.step(val_loss)
      if val_loss < lossMIN:
        print(
          "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epochID, lossMIN, val_loss,
                                                                                             save_model_path))
        lossMIN = val_loss
        patience_cnt = 0
        save_checkpoint({
          'epoch': epochID + 1,
          'lossMIN': lossMIN,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
        }, True, filename=save_model_path)
      else:
        patience_cnt += 1
      if patience_cnt > options.patience:
        break

    with open(log_file, 'a') as f:
      f.write(experiment + "\n")
      f.close()

output_file = os.path.join(output_path, options.exp_name + "_test.txt")
if options.mode == "train" or options.mode == "test":
  augment = Augmentation(normalize=options.normalization).get_augmentation(
    "{}_{}".format(options.augment, options.img_size), "test", options.test_augment)
  datasetTest = VinDrCXR(images_path=options.data_dir, file_path=options.test_list,
                                 augment=augment,in_channels=options.in_channels)
  test_data = DataLoader(dataset=datasetTest, batch_size=options.batch_size, shuffle=False,
                         num_workers=options.workers, pin_memory=True)

cudnn.benchmark = True
log_file = os.path.join(model_path, "models.log")
if not os.path.isfile(log_file):
  print("log_file ({}) not exists!".format(log_file))
else:
  mean_auc = []
  with open(log_file, 'r') as f_r, open(output_file, 'a') as f_w:
    exp_name = f_r.readline()
    while exp_name:
      exp_name = exp_name.replace('\n', '')
      pathCheckpoint = os.path.join(model_path, exp_name + ".pth.tar")
      y_test, p_test = test(pathCheckpoint, test_data, options)
      aurocIndividual = computeAUROC(y_test, p_test, options.num_class)
      print(">>{}: AUC = {}".format(exp_name, options.class_name))
      print(">>{}: AUC = {}".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator=',')))
      f_w.write("{}: AUC = {}\n".format(exp_name, options.class_name))
      f_w.write(
        "{}: AUC = {}\n".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator='\t')))
      aurocMean = np.array(aurocIndividual).mean()
      print(">>{}: AUC = {:.4f}".format(exp_name, aurocMean))
      f_w.write("{}: AUC = {:.4f}\n".format(exp_name, aurocMean))
      mean_auc.append(aurocMean)
      exp_name = f_r.readline()
    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    f_w.write("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
    print(">> All trials: mAUC(mean)  = {:.4f}".format(np.mean(mean_auc)))
    f_w.write("All trials: mAUC(mean)  = {:.4f}\n".format(np.mean(mean_auc)))
    print(">> All trials: mAUC(std)  = {:.4f}".format(np.std(mean_auc)))
    f_w.write("All trials: mAUC(std)  = {:.4f}\n".format(np.std(mean_auc)))

