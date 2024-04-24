from utils import *
import torch
import torch.nn as nn
import time

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, feature_similarities = model(im_q=images[0], im_k=images[1])
        if args.n == 0:
            loss = criterion(output, target)
        else:
            loss = criterion(output, target,feature_similarities,args )

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time,losses],
        prefix="Validation: ")

    # switch to train mode
    model.eval()
    end = time.time()
    for i, (images) in enumerate(val_loader):
        with torch.no_grad():
        # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

            output, target, feature_similarities = model(im_q=images[0], im_k=images[1])
            if args.n == 0:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target,feature_similarities,args )

            losses.update(loss.item(), images[0].size(0))

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


class PurposivePrunerLoss(nn.Module):
    def __init__(self,sim_threshold=0.8):
        super(PurposivePrunerLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.sim_threshold = sim_threshold

    def forward(self, output, target,cosine_similarities,args):
        losses = []
        for i in range(output.shape[0]):
            label = torch.zeros([1],dtype=torch.long).cuda(args.gpu)
            o=output[i].cuda(args.gpu)
            c=cosine_similarities[i].cuda(args.gpu)
            mask=c<self.sim_threshold
            gt = torch.tensor([True], dtype=torch.bool).cuda(args.gpu)
            mask = torch.cat((gt, mask), 0).cuda()
            masked_output = torch.masked_select(o, mask)
            masked_output = torch.unsqueeze(masked_output, 0)
            loss=self.criterion(masked_output,label)
            losses.append(loss)

        losses = torch.stack(losses).mean()
        return losses
