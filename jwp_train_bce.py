from gensim.models.doc2vec import Doc2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import Namespace
import numpy as np

d2vModel = Doc2Vec.load('d2vmodel/d2vmodel.model')

# train data 3k
POSTIVE_COMMENT_STRAT = 0
NEGATIVE_COMMENT_START = 4000
postiveAns = torch.ones([3000,1],dtype=torch.float)
negativeAns = torch.zeros([3000,1],dtype=torch.float)
postiveComments = []
negativeComments = []

for i in range(POSTIVE_COMMENT_STRAT, POSTIVE_COMMENT_STRAT + 3000):
    tmp = d2vModel.docvecs[str(i)]
    postiveComments.append(tmp)
postiveComments = torch.FloatTensor(postiveComments)

for i in range(NEGATIVE_COMMENT_START, NEGATIVE_COMMENT_START + 3000):
    tmp = d2vModel.docvecs[str(i)]
    negativeComments.append(tmp)
negativeComments = torch.FloatTensor(negativeComments)

trainData = torch.cat((postiveComments,negativeComments))
trainDataAns = torch.cat((postiveAns,negativeAns))

# test data 1k
T_POSTIVE_COMMENT_STRAT = 3000
T_NEGATIVE_COMMENT_START = 7000
t_postiveAns = torch.ones([1000,1],dtype=torch.float)
t_negativeAns = torch.zeros([1000,1],dtype=torch.float)
t_postiveComments = []
t_negativeComments = []

for i in range(T_POSTIVE_COMMENT_STRAT, T_POSTIVE_COMMENT_STRAT + 1000):
    tmp = d2vModel.docvecs[str(i)]
    t_postiveComments.append(tmp)
t_postiveComments = torch.FloatTensor(t_postiveComments)

for i in range(T_NEGATIVE_COMMENT_START, T_NEGATIVE_COMMENT_START + 1000):
    tmp = d2vModel.docvecs[str(i)]
    t_negativeComments.append(tmp)
t_negativeComments = torch.FloatTensor(t_negativeComments)

testData = torch.cat((t_postiveComments,t_negativeComments))
testDataAns = torch.cat((t_postiveAns,t_negativeAns))

class JWP(nn.Module):
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(JWP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        
    def forward(self, x):
        x = F.relu(self.hidden(x).squeeze())
        x = F.relu(self.hidden2(x).squeeze())
        x = F.sigmoid(self.out(x))

        return x

lr = 0.004
min_lr = 0.0005
def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch % 5 == 0 and epoch != 0:
        lr = lr * 0.9
        if(lr < min_lr):
            lr = min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    

if __name__ == "__main__":
    net = JWP(200,150,100,1) 
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()  # the target label is NOT an one-hotted

    for t in range(30):
        adjust_learning_rate(optimizer,t)
        # 打亂資料
        torch.manual_seed(t)
        trainData=trainData[torch.randperm(trainData.size()[0])]
        torch.manual_seed(t)
        trainDataAns=trainDataAns[torch.randperm(trainDataAns.size()[0])]
        # train
        net.train()
        out = net(trainData)
        outAsAns = out.clone().detach().numpy()
        outAsAns = np.where([outAsAns > 0.5],1.0,0.0)
        
        loss = loss_func(out,trainDataAns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # eval
        net.eval()
        out = net(testData)
        t_outAsAns = out.clone().detach().numpy()
        t_outAsAns = np.where([t_outAsAns > 0.5],1.0,0.0)
        
        print(
            "R:",t ,
            "loss:",round(loss.item(),3),
            "train_acc:",round(np.mean(outAsAns == trainDataAns.numpy()),3),
            "test_acc:",round(np.mean(t_outAsAns == testDataAns.numpy()),3),
            "LR:",lr
        )

    torch.save(net, 'torchmodel/pytorch_bce.model')
    print('model save')
