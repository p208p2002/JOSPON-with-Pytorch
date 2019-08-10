import torch
from jwp_train import JWP
from gensim.models.doc2vec import Doc2Vec
import jieba
import torch.nn.functional as F
import torch
import numpy as np

modelSelect = int(input("[1]:BCE MODEL, [2]:CE MODEL\n"))

print("init...")

# jieba初始化
jieba.set_dictionary('dict/dict.txt.big')
jieba.load_userdict('dict/my_dict')
jieba.initialize()

#d2v
model= Doc2Vec.load("d2vmodel/d2vmodel.model")

# load pytorch
if(modelSelect == 1):
    net = torch.load('torchmodel/pytorch_bce.model')
elif(modelSelect == 2):
    net = torch.load('torchmodel/pytorch_ce.model')
net.eval()

# test_data
while True:
    ts = input("輸入評價:")
    test_data = list(jieba.cut(ts))
    v1 = torch.zeros(200)
    for i in range(50):
        v1 = v1 + torch.tensor(model.infer_vector(test_data))
    v1 = v1 / 50
    res = net(torch.tensor(v1))
    out = res
    res = res.clone().detach().numpy()[0]
    print(out)

    if(res>0.5):
        print("正面")
    else:
        print("反面")