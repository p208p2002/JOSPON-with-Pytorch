from gensim import models
model = models.Word2Vec.load('w2vmodel/word2vec.model')
print(model['測試'])
print(len(model['測試']))