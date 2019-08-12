# -*- coding: UTF-8 -*-
"""
PyTorch類神經網路模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class JWP(nn.Module):
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(JWP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        
    def forward(self, x, apply_sigmoid=False):
        x = F.relu(self.hidden(x).squeeze())
        x = F.relu(self.hidden2(x).squeeze())
        # 
        if(apply_sigmoid):
            x = torch.sigmoid(self.out(x))
        else:
            x = self.out(x)

        return x