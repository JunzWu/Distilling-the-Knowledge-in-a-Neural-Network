import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from adversary import adversarial_eval

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-t',
                    '--temperature',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='temperature (default: 1.0)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=8,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--wt-dis',
                    default=0.5,
                    type=float,
                    metavar='W',
                    help='weight of distillation_model (default: 0.5)')
parser.add_argument('--wt-teacher',
                    default=0.5,
                    type=float,
                    metavar='W',
                    help='weight of teacher (default: 0.5)')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')

TOTAL_CLASSES = 1000

logits = []


def fetch_logits(self, input, output):
    global logits
    logits = output


class Hooker_for_logits:
    def __init__(self, model):
        self.model = model
        self.model.fc.register_forward_hook(fetch_logits)

    def forward(self, train_input):
        output = self.model(train_input)
        return output, logits


class Distillation(nn.Module):
    def __init__(self):
        super(Distillation, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batch_norm = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

        self.fc = nn.Linear(64 * 52 * 52, TOTAL_CLASSES)

    def forward(self, x):
        x = self.ReLU(self.batch_norm(self.conv1(x)))
        x = self.pool(self.conv2(x))
        x = self.ReLU(self.batch_norm(self.conv2(x)))
        x = self.pool(self.conv2(x))
        x = self.ReLU(self.batch_norm(self.conv2(x)))
        x = self.conv3(x)
        x = self.ReLU(self.batch_norm(self.conv3(x)))
        x = self.drop(self.ReLU(self.batch_norm(self.conv3(x))))

        x = x.view(-1, 64 * 52 * 52)
        x = self.fc(x)

        return x

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Small_MNIST_Net(nn.Module):
    def __init__(self):
        super(Small_MNIST_Net, self).__init__()
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Simply call main_worker function
    main_worker(args)

cuda1 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')

def main_worker(args):
    # create model
    global teacher_model
    teacher_model = torch.hub.load('pytorch/vision',
                                   'resnet50',
                                   pretrained=True)
    teacher_model = teacher_model.to(cuda1)
    # teacher_model = MNIST_Net().to(cuda1)
    # PATH = './cifar_net.pth'
    # teacher_model.load_state_dict(torch.load(PATH))

    global dis_model
    # dis_model = Distillation()
    # dis_model = dis_model.to(cuda1)

    dis_model = torch.hub.load('pytorch/vision',
                               'resnet18',
                               pretrained=False)
    dis_model = dis_model.to(cuda1)
    # dis_model = MNIST_Net().to(cuda1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # dis_optimizer = torch.optim.SGD(dis_model.parameters(),
    #                                 args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)

    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=args.lr)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(True),
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=int(args.batch_size/2))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.batch_size/2))
    '''

    '''
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=int(args.batch_size/2), pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=int(args.batch_size/2), pin_memory=True)
    '''

    writer = SummaryWriter()

    for epoch in range(args.epochs):
        adjust_learning_rate(dis_optimizer, epoch, args)

        # train for one epoch
        train(train_loader, teacher_model, dis_model, criterion, dis_optimizer,
              epoch, args, writer)

        # evaluate on validation set
        validate(val_loader, dis_model, criterion, args, writer, epoch)

    adversarial_eval(dis_model, device, train_loader)

def train(train_loader, teacher_model, dis_model, criterion, optimizer, epoch,
          args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    dis_model.train()

    teacher_hooker = Hooker_for_logits(teacher_model)
    dis_hooker = Hooker_for_logits(dis_model)

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(cuda1)
        targets = targets.to(cuda1)

        # compute output
        dis_outputs, dis_logits = dis_hooker.forward(images)
        soft_log_probs = F.log_softmax(dis_logits / args.temperature, dim=1)
        dis_loss = criterion(dis_outputs, targets)
        # print('dis_loss: ', dis_loss)

        _, teacher_logits = teacher_hooker.forward(images)
        soft_targets = (args.temperature ** 2) * F.softmax(teacher_logits / args.temperature, dim=1)
        teacher_loss = F.kl_div(soft_log_probs,
                                soft_targets.detach(),
                                size_average=False) / soft_targets.shape[0]
        # print('teacher_loss: ', teacher_loss)

        loss = args.wt_dis * dis_loss + args.wt_teacher * teacher_loss
        # loss = dis_loss
        # print('loss: ', loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(dis_outputs, targets, topk=(1, 5))
        losses.update(loss, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    writer.add_scalar('Acc@1/train', top1.avg, epoch)
    writer.add_scalar('Acc@5/train', top5.avg, epoch)

    print('Train * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                     top5=top5))

def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(cuda1)
            targets = targets.to(cuda1)

            # compute output
            outputs = model(images)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        writer.add_scalar('Acc@1/val', top1.avg, epoch)
        writer.add_scalar('Acc@5/val', top5.avg, epoch)

        # TODO: this should also be done with the ProgressMeter
        print('Test * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                        top5=top5))

    return top1.avg


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
