#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class TerminatorNet(nn.Module):
  def __init__(self, shape):
    super(TerminatorNet, self).__init__()
    self.net = nn.Linear(shape, 2)

  def forward(self, input):
    return self.net(input)


class ArgumentNet(nn.Module):
  def __init__(self, shape, hidden_size, num_args):
    super(ArgumentNet, self).__init__()
    self.num_args = num_args
    self.nets = [ nn.Linear(shape, hidden_size) for x in range(num_args) ]

  def forward(self, input):
    return [ n(input) for n in self.nets ]
