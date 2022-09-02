'''
Training script for Fashion-MNIST
https://github.com/zhunzhong07/Random-Erasing
'''
from __future__ import print_function
import math
import argparse
import os
import shutil
import time
import random
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import transforms
import numpy as np
from emnist import extract_training_samples, extract_test_samples
from dataset import CustomDataset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torchattacks import *

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

noise_levels = [1.0, 3.0, 5.0, 10.]
# noise_levels = [0.1, ]

parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['fashionmnist', 'emnist', 'cifar10'],
                    help='fashionmnist or emnist')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--k', default=5, type=int, help='top k accuracy')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--experiment_name', default='exp', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to specified checkpoint (default: none) '
                         'e.g. checkpoint/fashionmnist/{args.arch}_D{args.depth}_W{args.widen_factor}/{args.manualSeed}. '
                         'useful for 1. preemption: it will load the checkpoint.pth.tar and resume training, '
                         '2. when evaluation==True, it will load '
                         'model_best.pth.tar and do the evaluation ')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn',
                    choices=['resnet', 'wrn'])
parser.add_argument('--depth', type=int, default=10, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Random Erasing
parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

#attack method
parser.add_argument('--attack_method', type=str, default='PGD',
                    help='class name of the attack method')
parser.add_argument('--attack_param', type=str, default="{}",
                    help="the parameters for the attack method. eg. "
                         "{'eps':0.1,} (no space in between)")
parser.add_argument('--evaluate_frequency', type=int, default=50,
                    help='how many epochs to evaluate the robustness. '
                         'Default evaluate every 50 epochs ')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
global checkpoint_path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.attack_param = eval(args.attack_param)
# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    checkpoint_path = f"{args.checkpoint}/{args.experiment_name}/{args.dataset}/" \
                      f"{args.arch}_D{args.depth}_W{args.widen_factor}_{args.attack_method}/" \
                      f"{str(args.manualSeed)}"
    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    if args.dataset == 'fashionmnist':
        # Data
        import models.fashion as models
        print('==> Preparing dataset %s' % args.dataset)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1,
                                     mean=[0.4914]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataloader = datasets.FashionMNIST
        num_classes = 10
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'emnist':
        import models.fashion as models
        print('==> Preparing dataset %s' % args.dataset)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1,
                                     mean=[0.4914]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        # https://pypi.org/project/emnist/
        train_images, train_labels = extract_training_samples('byclass')
        test_images, test_labels = extract_test_samples('byclass')
        # train_images = train_images / 255.0
        # test_images = test_images / 255.0
        num_classes = 62
        train_dataset = CustomDataset(train_images, torch.LongTensor(train_labels),
                                      transform_train)
        val_dataset = CustomDataset(test_images, torch.LongTensor(test_labels), transform_test)
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.train_batch,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.test_batch,
                                                 shuffle=False)
    elif args.dataset == 'cifar10':
        import models.cifar as models
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(probability=args.p, sh=args.sh,
                                     r1=args.r1, ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataloader = datasets.CIFAR10
        num_classes = 10

        trainset = dataloader(root='./data', train=True, download=True,
                              transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch,
                                      shuffle=True, num_workers=args.workers)

        testset = dataloader(root='./data', train=False, download=False,
                             transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch,
                                     shuffle=False, num_workers=args.workers)
    else:
        raise NotImplementedError('unrecognized dataset name')

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )

    model = torch.nn.DataParallel(model).to(DEVICE)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    attack_method = eval(args.attack_method)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = checkpoint_path
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        try:
            assert os.path.isfile(f'{args.resume}/checkpoint.pth.tar'), 'Error: no checkpoint directory found!'
            checkpoint_path = os.path.dirname(args.resume)
            checkpoint = torch.load(f'{args.resume}/checkpoint.pth.tar',
                                    map_location=DEVICE)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(checkpoint_path, 'log.txt'),
                            title=title, resume=True)
        except AssertionError:
            if os.path.isfile(f'{checkpoint_path}/checkpoint.pth.tar'):
                print(
                    f'==> Resume dir incorrect.. '
                    f'loading from default checkpoint_path: {checkpoint_path}')
                checkpoint = torch.load(f'{checkpoint_path}/checkpoint.pth.tar',
                                        map_location=DEVICE)
                best_acc = checkpoint['best_acc']
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger = Logger(os.path.join(checkpoint_path, 'log.txt'),
                                title=title, resume=True)
            else:
                print('no checkpoints to resume from. Start a new run')
                print('models stored at ', checkpoint_path)
                logger = Logger(os.path.join(checkpoint_path, 'log.txt'),
                                title=title)
                names = ['epoch', 'Learning Rate', 'Train Loss', 'Valid Loss',
                         'Train Acc.', 'Valid Acc.',
                         'Train topk acc.', 'Valid topk acc.']
                for nl in noise_levels:
                    names += [f'rtest_loss-{nl}', f'rtest_acc-{nl}',
                              f'rtest_acc_topk-{nl}']
                logger.set_names(names)
                with open(os.path.join(checkpoint_path, 'train_result.csv'), 'w') as f:
                    write = csv.writer(f)
                    write.writerow(names)

    elif args.evaluate:
        # Load checkpoint.
        print('==> evaluating from best model..')
        assert os.path.isfile(
            f'{args.resume}/model_best.pth.tar'), 'Error: no checkpoint directory found!'
        checkpoint_path = os.path.dirname(args.resume)
        checkpoint = torch.load(f'{args.resume}/model_best.pth.tar',
                                map_location=DEVICE)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title,
                        resume=True)

        print('\nEvaluation only')
        test_loss, test_acc, test_acc_topk = test(testloader, model, criterion, start_epoch,
                                                    use_cuda, args.k)
        print(' Test Loss:  %.8f, Test Acc:  %.2f, Test Acc Topk:  %.2f' % (test_loss, test_acc, test_acc_topk))
        result = [test_loss, test_acc, test_acc_topk]
        names = ['test_loss', 'test_acc', 'test_acc_topk']
        for nl in noise_levels:
            attack_param = args.attack_param
            attack_param['eps'] = nl
            rtest_loss, rtest_acc, rtest_acc_topk = robustness_test(testloader,
                                                                    model,
                                                                    criterion,
                                                                    start_epoch,
                                                                    use_cuda,
                                                                    args.k,
                                                                    attack_method,
                                                                    args.attack_param)
            result += [rtest_loss, rtest_acc, rtest_acc_topk]
            names += [f'rtest_loss-{nl}', f'rtest_acc-{nl}', f'rtest_acc_topk-{nl}']
            print(f'Robustness test {nl}'+ ': Loss:  %.8f, Acc:  %.2f, Acc Topk:  %.2f' % (
            rtest_loss, rtest_acc, rtest_acc_topk))
        with open(f'{checkpoint_path}/test_result.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(names)
            write.writerow(result)

        return

    else:
        print('models stored at ', checkpoint_path)
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title)
        names = ['epoch', 'Learning Rate', 'Train Loss', 'Valid Loss',
                          'Train Acc.', 'Valid Acc.',
                          'Train topk acc.', 'Valid topk acc.']
        for nl in noise_levels:
            names += [f'rtest_loss-{nl}', f'rtest_acc-{nl}', f'rtest_acc_topk-{nl}']
        logger.set_names(names)
        with open(os.path.join(checkpoint_path, 'train_result.csv'), 'w') as f:
            write = csv.writer(f)
            write.writerow(names)


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc, train_topk_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, args.k)
        test_loss, test_acc, test_topk_acc = test(testloader, model, criterion, epoch, use_cuda, args.k)
        print(f'Time elapsed: {(time.time()-start_time): .3f} seconds')
        print(f'training loss: {train_loss: .3f}, train acc: {train_acc: .3f}, test loss: {test_loss: .3f}, '
              f'test acc: {test_acc: .3f}')
        attack_results = [epoch+1, state['lr'], train_loss, test_loss, train_acc, test_acc,
                       train_topk_acc, test_topk_acc]
        for nl in noise_levels:
            attack_param = args.attack_param
            attack_param['eps'] = nl
            if (epoch+1) % args.evaluate_frequency == 0:
                rtest_loss, rtest_acc, rtest_acc_topk = robustness_test(testloader,
                                                                        model,
                                                                        criterion,
                                                                        start_epoch,
                                                                        use_cuda,
                                                                        args.k,
                                                                        attack_method,
                                                                        args.attack_param)
                attack_results += [rtest_loss, rtest_acc, rtest_acc_topk]
            else:
                attack_results += [math.nan, math.nan, math.nan]

        # append logger file
        logger.append(attack_results)
        with open(os.path.join(checkpoint_path, 'train_result.csv'), 'a') as f:
            write = csv.writer(f)
            write.writerow(attack_results)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_path, 'log.eps'))

    print('Best acc:')
    print(best_acc)

    # test_loss, test_acc, test_acc_topk = test(testloader, model, criterion,
    #                                           start_epoch,
    #                                           use_cuda, args.k)
    # print(' Test Loss:  %.8f, Test Acc:  %.2f, Test Acc Topk:  %.2f' % (
    # test_loss, test_acc, test_acc_topk))
    #
    # rtest_loss, rtest_acc, rtest_acc_topk = robustness_test(testloader, model,
    #                                                         criterion,
    #                                                         start_epoch,
    #                                                         use_cuda, args.k,
    #                                                         attack_method,
    #                                                         args.attack_param)
    # print('Robustness test: Loss:  %.8f, Acc:  %.2f, Acc Topk:  %.2f' % (
    #     rtest_loss, rtest_acc, rtest_acc_topk))

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, k):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, k))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)

def test(testloader, model, criterion, epoch, use_cuda, k):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, requires_grad=False), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, k))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)


def robustness_test(testloader, model, criterion, epoch, use_cuda, k, attack_method=None, attack_param={}):

    if attack_method is None:
        print('no attack_method')
        return

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    atk = attack_method(model, **attack_param)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = atk(inputs, targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, requires_grad=False), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, k))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
