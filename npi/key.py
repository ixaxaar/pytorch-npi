#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Unsqueeze(nn.Module):
  def __init__(self, ndim):
    super(Unsqueeze, self).__init__()
    self.ndim = ndim

  def forward(self, input):
    return input.unsqueeze(self.ndim)


class Tile(nn.Module):
  def __init__(self, max_size, dim):
    super(Tile, self).__init__()
    self.max_size = max_size
    self.dim = dim

  def forward(self, input):
    return input.repeat(*[ self.max_size if x == self.dim else 1 for x in range(len(input.shape)) ])


class KeyNet(nn.Module):
  def __init__(self, shape, hidden_size, program_memory_cells):
    super(KeyNet, self).__init__()
    self.program_memory_cells = program_memory_cells

    self.net = nn.Sequential(
      nn.Linear(shape, hidden_size),
      nn.Linear(hidden_size, hidden_size),
      Unsqueeze(1),
      Tile(program_memory_cells, dim=1)
    )

  def forward(self, input, program_memory):
    return (self.net(input) * program_memory).sum(dim=-1)
