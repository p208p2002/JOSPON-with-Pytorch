import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace
import numpy as np
import pickle
from JWP import JWP

with open('dataset/waimai_10k_tw.pkl','rb') as f:
    waimai10k = pickle.load(f)

"""
訓練資料3K
"""
POSTIVE_COMMENT_STRAT = 0
NEGATIVE_COMMENT_START = 4000
postiveAns = torch.ones([3000,1],dtype=torch.float)
negativeAns = torch.zeros([3000,1],dtype=torch.float)
postiveComments = []
negativeComments = []

for i in range(POSTIVE_COMMENT_STRAT, POSTIVE_COMMENT_STRAT + 3000):
    vec,ans = waimai10k[str(i)]
    postiveComments.append(vec)
postiveComments = torch.FloatTensor(postiveComments)

for i in range(NEGATIVE_COMMENT_START, NEGATIVE_COMMENT_START + 3000):
    vec,ans = waimai10k[str(i)]
    negativeComments.append(vec)
negativeComments = torch.FloatTensor(negativeComments)

trainData = torch.cat((postiveComments,negativeComments))
trainDataAns = torch.cat((postiveAns,negativeAns))

"""
測試資料 1K
"""
T_POSTIVE_COMMENT_STRAT = 3000
T_NEGATIVE_COMMENT_START = 7000
t_postiveAns = torch.ones([1000,1],dtype=torch.float)
t_negativeAns = torch.zeros([1000,1],dtype=torch.float)
t_postiveComments = []
t_negativeComments = []

for i in range(T_POSTIVE_COMMENT_STRAT, T_POSTIVE_COMMENT_STRAT + 1000):
    vec,ans = waimai10k[str(i)]
    t_postiveComments.append(vec)
t_postiveComments = torch.FloatTensor(t_postiveComments)

for i in range(T_NEGATIVE_COMMENT_START, T_NEGATIVE_COMMENT_START + 1000):
    vec,ans = waimai10k[str(i)]
    t_negativeComments.append(vec)
t_negativeComments = torch.FloatTensor(t_negativeComments)

testData = torch.cat((t_postiveComments,t_negativeComments))
testDataAns = torch.cat((t_postiveAns,t_negativeAns))

lr = 0.009
min_lr = 0.001
def adjust_learning_rate(optimizer, epoch):
    """
    調整學習率
    """
    global lr
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.65
        if(lr < min_lr):
            lr = min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100
    

if __name__ == "__main__":
    EARLY_STOP_LOSS = 0.35
    EPOCH = 100

    net = JWP(200,150,100,1)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for t in range(EPOCH):
        adjust_learning_rate(optimizer,t)

        """
        打亂資料
        """
        torch.manual_seed(t)
        trainData=trainData[torch.randperm(trainData.size()[0])]
        torch.manual_seed(t)
        trainDataAns=trainDataAns[torch.randperm(trainDataAns.size()[0])]
        
        """
        Train phase
        """
        net.train()
        optimizer.zero_grad()
        out = net(trainData)
        trainAcc = compute_accuracy(out,trainDataAns.long())
        loss = loss_func(out,trainDataAns)        
        loss.backward()
        optimizer.step()

        """
        Eval phase
        """
        net.eval()
        t_out = net(testData)
        testAcc = compute_accuracy(t_out,testDataAns.long())
        t_loss = loss_func(t_out,testDataAns)
        
        """
        Result
        """
        print(
            "epoch:",t+1 ,
            "train_loss:",round(loss.item(),3),
            "train_acc:",round(trainAcc,3),
            "test_loss:",round(t_loss.item(),3),
            "test_acc:",round(testAcc,3),
            "LR:",lr
        )

        if(t_loss <= EARLY_STOP_LOSS):
            print("Early stop")
            break
    
    torch.save(net, 'torchmodel/pytorch_bce.model')
    print('model save')
