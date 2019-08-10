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

postiveAns = torch.tensor(0)
postiveAns = postiveAns.repeat(3000)

negativeAns = torch.tensor(1)
negativeAns = negativeAns.repeat(3000)

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

t_postiveAns = torch.tensor(0)
t_postiveAns = t_postiveAns.repeat(1000)

t_negativeAns = torch.tensor(1)
t_negativeAns = t_negativeAns.repeat(1000)

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
    net = JWP(200,150,100,2) 
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    num_correct = 0
    num_samples = 0
    t_num_correct = 0
    t_num_samples = 0
    for t in range(30):

        # train
        net.train()
        out = net(trainData)

        _, preds = out.data.cpu().max(1)
        num_correct += (preds == trainDataAns).sum()
        num_samples += preds.size(0)
        acc = float(num_correct) / num_samples

        loss = loss_func(out,trainDataAns)
        optimizer.zero_grad()
        loss.backward()
        adjust_learning_rate(optimizer,t)
        optimizer.step()

        # eval
        net.eval()
        t_out = net(testData)

        _, t_preds = t_out.data.cpu().max(1)
        t_num_correct += (t_preds == testDataAns).sum()
        t_num_samples += t_preds.size(0)
        t_acc = float(t_num_correct) / t_num_samples
        
        print(
            "R:",t ,
            "loss:",round(loss.item(),3),
            "train_acc:",round(acc,3),
            "test_acc:",round(t_acc,3),
            "LR:",lr
        )

    torch.save(net, 'torchmodel/pytorch_ce.model')
    print('model save')
