#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .core import NPICore


class NPI(nn.Module):
  def __init__(
      self,
      encoder,
      input_size,
      hidden_size,
      num_layers=2,
      kind='lstm',
      dropout=0,
      bidirectional=False,
      num_args=10,
      programs=1000,
      **kwargs
  ):
    super(NPI, self).__init__()

    # networks
    self.encoder = encoder
    self.core_net = NPICore(input_size, hidden_size, num_layers, kind, dropout, bidirectional, **kwargs)
    self.terminator_net = TerminatorNet(input_size * hidden_size)
    self.argument_net = ArgumentNet(input_size * hidden_size, hidden_size, num_args)
    self.key_net = KeyNet(input_size * hidden_size, hidden_size, programs)

    # variables
    self.memory =



