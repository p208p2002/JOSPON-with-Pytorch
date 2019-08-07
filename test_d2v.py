from gensim.models.doc2vec import Doc2Vec
def test_doc2vec():
    # 載入模型
    model = Doc2Vec.load('d2vmodel/d2vmodel.model')  # you can continue training with the loaded model!
    # 與標籤‘0’最相似的
    # print(model.docvecs.most_similar("2"))
    # 進行相關性比較
    print(model.docvecs.similarity('0','11885'))
    # 輸出標籤為‘10’句子的向量
    # print(model.docvecs['10'])
    # 也可以推斷一個句向量(未出現在語料中)
    # words = "你好棒"
    # print(model.infer_vector([words]))
    

test_doc2vec()