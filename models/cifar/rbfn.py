from random import random

import torch
import torch.nn as nn

torch.manual_seed(777)

'''RbfNet for cifar dataset. 
Ported form 
https://github.com/csnstat/rbfn
'''

__all__ = ['rbfn']


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class RbfNet(nn.Module):
    def __init__(self, num_centers, num_classes=10):
        super(RbfNet, self).__init__()
        self.num_class = num_classes
        self.num_centers = num_centers
        centers = torch.rand(num_centers, 28 * 28)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers) / 10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)
        initialize_weights(self)

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False).sqrt()))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return class_score


def rbfn(**kwargs):
    """
    Constructs a RBFN model.
    """
    return RbfNet(**kwargs)
