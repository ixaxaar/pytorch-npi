#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from dnc import DNC, SAM, SDNC


class NPICore(nn.Module):
  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers=2,
      kind='lstm',
      dropout=0,
      bidirectional=False
  ):
    super(NPICore, self).__init__()
    if kind == 'lstm':
      self.net = nn.LSTM(input_size, hidden_size,
                         num_layers=num_layers, batch_first=True,
                         dropout=dropout, bidirectional=bidirectional)

  def forward(self, state, program, hidden=None):
    state = state.unsqueeze(1)
    concated = T.cat([state, program], dim=-1)

    output, hidden = self.net(concated, hidden)

    return output, hidden
