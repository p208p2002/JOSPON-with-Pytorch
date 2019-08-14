import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
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
trainDataSet = Data.TensorDataset(trainData, trainDataAns)

trainDataLoader = Data.DataLoader(
    dataset = trainDataSet,
    batch_size = 200,
    shuffle = True,
    num_workers = 4
)

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
testDataSet = Data.TensorDataset(testData, testDataAns)

testDataLoader = Data.DataLoader(
    dataset = testDataSet,
    batch_size = 200,
    shuffle = True,
    num_workers = 4
)

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
    EPOCH = 100
    net = JWP(200,150,100,1)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for t in range(EPOCH):
        adjust_learning_rate(optimizer,t)
        TrainAcc = 0.0
        TrainLoss = 0.0
        for step,(batchData, batchTarget) in enumerate(trainDataLoader):
            """
            Train phase
            """
            net.train()
            optimizer.zero_grad()
            out = net(batchData)
            trainAcc = compute_accuracy(out,batchTarget.long())
            TrainAcc = TrainAcc + trainAcc
            loss = loss_func(out,batchTarget)
            TrainLoss = TrainLoss + loss
            loss.backward()
            optimizer.step()
        TrainLoss = TrainLoss / (step+1)
        TrainAcc = TrainAcc / (step+1)
        
        TestAcc = 0.0
        TestLoss = 0.0
        for step,(t_batchData, t_batchTarget) in enumerate(trainDataLoader):
            """
            Eval phase
            """
            net.eval()
            t_out = net(t_batchData)
            testAcc = compute_accuracy(t_out,t_batchTarget.long())
            TestAcc = TestAcc + testAcc
            t_loss = loss_func(t_out,t_batchTarget)
            TestLoss = TestLoss + t_loss
        TestLoss = TestLoss / (step+1)
        TestAcc = TestAcc / (step+1)
        
        """
        Result
        """
        print(
            "epoch:",t+1 ,
            "train_loss:",round(TrainLoss.item(),3),
            "train_acc:",round(TrainAcc,3),
            "test_loss:",round(TestLoss.item(),3),
            "test_acc:",round(TestAcc,3),
            "LR:",lr
        )
    
    torch.save(net, 'torchmodel/pytorch_bce.model')
    print('model save')
