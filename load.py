from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load('d2vmodel/d2vmodel.model')  # you can continue training with the loaded model!
# model.infer_vector(['你真的好棒']) # 推斷新的向量