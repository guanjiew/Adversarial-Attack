import numpy as np
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from emnist import extract_test_samples
from dataset import CustomDataset
import torchattacks
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import models.fashion as models
from helper import imshow, image_folder_custom_label

import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

# Checkpoints
parser.add_argument('--dataset', default='e-checkpoint', type=str,
                    help='emnist or fashionmnist')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

if args.dataset == 'emnist':
    # https://pypi.org/project/emnist/
    test_images, test_labels = extract_test_samples('byclass')
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0
    num_classes = 62
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
    val_dataset = CustomDataset(test_images, test_labels, transform)
    testloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
    checkpoint_dir = 'echeckpoint/model_best.pth.tar'
elif args.dataset == 'fashionmnist':
    dataloader = datasets.FashionMNIST
    num_classes = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    testset = dataloader(root='./data', train=False, download=True,
                         transform=transform)
    testloader = data.DataLoader(testset, batch_size=args.test_batch,
                                 shuffle=False, num_workers=args.workers)
    checkpoint_dir = 'fcheckpoint/model_best.pth.tar'

model = torch.load(checkpoint_dir, map_location=DEVICE)



