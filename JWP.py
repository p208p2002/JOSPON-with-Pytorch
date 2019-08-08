from gensim.models.doc2vec import Doc2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

d2vModel = Doc2Vec.load('d2vmodel/d2vmodel.model')
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
# postiveData = torch.cat((postiveComments,postiveAns), ).type(torch.FloatTensor)
# negativeData = torch.cat((negativeComments,negativeAns), ).type(torch.FloatTensor)


class JWP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(JWP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.sigmoid(self.hidden(x).squeeze())
        x = self.out(x)
        return x

if __name__ == "__main__":
    net = JWP(200,150,1)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    loss_func = torch.nn.BCEWithLogitsLoss()  # the target label is NOT an one-hotted


    #
    running_loss = 0.0
    running_acc = 0.0
    for t in range(150):
        # out = net(postiveComments)
        out = net(trainData)
        # print(F.sigmoid(out[0]),trainDataAns[0],F.sigmoid(out[5000]),trainDataAns[5000])
        loss = loss_func(out,trainDataAns)
        loss.backward()
        optimizer.step()
        
        # loss_batch = loss.item()
        # running_loss += (loss_batch - running_loss) / (t+1)
        print(loss.data,F.sigmoid(out[t]),trainDataAns[t],F.sigmoid(out[t+3000]),trainDataAns[t+3000])
        # print('loss',)

        if(loss.data.numpy() <= 0.0001):
            break
        
        # if t % 2 == 0:
            # plot and show learning process


    torch.save(net, 'pytorch.model')
    print('model save')
