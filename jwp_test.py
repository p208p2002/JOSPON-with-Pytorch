import torch
from JWP import JWP
from gensim.models.doc2vec import Doc2Vec
import jieba
import torch.nn.functional as F
import torch
import numpy as np
from w2v_sentence_vector import W2VS

print("init...")
w2vs = W2VS()
modelSelect = int(input("[1]:BCE MODEL, [2]:CE MODEL\n"))

# test_data
while True:
    ts = input("輸入評價:")
    v1 = w2vs.getSenVec(ts)
    res = net(torch.FloatTensor(v1))
    out = res
    res = res.clone().detach().numpy()[0]
    print(round(res,3))

    if(res>0.5):
        print("正面")
    else:
        print("反面")