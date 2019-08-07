from gensim.models.doc2vec import Doc2Vec
import jieba

# jieba初始化
jieba.set_dictionary('dict/dict.txt.big')
jieba.load_userdict('dict/my_dict')
jieba.initialize()

model= Doc2Vec.load("d2vmodel/d2vmodel.model")
#to find the vector of a document which is not in training data
test_data = list(jieba.cut("送餐速度非常快，服務極好"))
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])