import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from argparse import Namespace
import numpy as np
import pickle
from JWP import JWP
from argparse import Namespace

args = Namespace(
    dataset_file = 'dataset/waimai_10k_tw.pkl',
    model_save_path='torchmodel/pytorch_bce.model',
    # Training hyper parameters
    batch_size = 200,
    learning_rate = 0.009,
    min_learning_rate = 0.001,
    num_epochs=50,
)

with open(args.dataset_file,'rb') as f:
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
    batch_size = args.batch_size,
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
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = 4
)

lr = args.learning_rate
min_lr = args.min_learning_rate
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
    """
    計算正確率
    """
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100
    

if __name__ == "__main__":
    EPOCH = args.num_epochs
    net = JWP(200,150,100,1)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for t in range(EPOCH):
        """
        動態調整學習率
        """
        adjust_learning_rate(optimizer,t)

        """
        Train phase
        """
        net.train() # 訓練模式        
        TrainAcc = 0.0
        TrainLoss = 0.0
        # Train batch
        for step,(batchData, batchTarget) in enumerate(trainDataLoader):
            optimizer.zero_grad() # 梯度歸零
            out = net(batchData)
            trainAcc = compute_accuracy(out,batchTarget.long()) # 取得正確率
            TrainAcc = TrainAcc + trainAcc
            loss = loss_func(out,batchTarget) # loss 計算
            TrainLoss = TrainLoss + loss
            loss.backward() # 反向傳播
            optimizer.step() # 更新權重
        TrainLoss = TrainLoss / (step+1) # epoch loss
        TrainAcc = TrainAcc / (step+1) # epoch acc

        """
        Eval phase
        """
        net.eval()
        TestAcc = 0.0
        TestLoss = 0.0
        # Eval batch
        for step,(t_batchData, t_batchTarget) in enumerate(trainDataLoader):            
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
    
    torch.save(net, args.model_save_path)
    print('model save')
