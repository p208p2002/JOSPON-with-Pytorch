from gensim.models.doc2vec import Doc2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import Namespace
import numpy as np

d2vModel = Doc2Vec.load('d2vmodel/d2vmodel.model')

class JWP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(JWP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x).squeeze())
        x = F.sigmoid(self.out(x))

        return x

def getSentense(tag):
    global d2vModel
    return d2vModel.docvecs[str(tag)]

POSTIVE_COMMENT_STRAT = 0
NEGATIVE_COMMENT_START = 4000
if __name__ == "__main__":
    print(getSentense(1),len(getSentense(1)))
    net = JWP(200,150,1)
    net.train()