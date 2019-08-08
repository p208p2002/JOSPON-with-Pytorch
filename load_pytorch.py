import torch
from JWP import JWP
from gensim.models.doc2vec import Doc2Vec
import jieba
import torch.nn.functional as F

# jieba初始化
jieba.set_dictionary('dict/dict.txt.big')
jieba.load_userdict('dict/my_dict')
jieba.initialize()

#d2v
model= Doc2Vec.load("d2vmodel/d2vmodel.model")

# load pytorch
# net = JWP(200,150,1)
net = torch.load('pytorch.model')

# test_data
test_data = list(jieba.cut("送餐時間過久,飯都涼了"))
# v1 = model.infer_vector(test_data)
# v1 = [v1]
v1 = [model.docvecs['2401']]
res = net(torch.tensor(v1))
res = F.sigmoid(res).detach().numpy()[0]
print(res)
if(res>0.5):
    print("正面")
else:
    print("反面")