# -*- coding: UTF-8 -*-
"""
使用預訓練的w2v模型取得句子向量
default size : 200
"""

from gensim import models
import jieba
import numpy as np
class W2VS():
    def __init__(self):
        """
        初始化加載
        """
        jieba.set_dictionary('dict/dict.txt.big')
        jieba.load_userdict('dict/my_dict')
        jieba.initialize()
        self.model = models.Word2Vec.load('w2vmodel/word2vec.model')
    
    def getSenVec(self,sentence):
        """
        取得單詞向量
        單詞向量相加平均
        返回向量(句子向量)
        """
        senCut = list(jieba.cut(sentence))
        lenOfCut = len(senCut)
        vecSum = np.zeros(200)
        for i in senCut:
            try:
                vec = self.model.wv.__getitem__(i)
                vecSum = np.add(vecSum, vec)
            except Exception as e:
                # print(e)
                lenOfCut -= 1
                continue
        if(lenOfCut == 0 ):
            return np.array([0]*200)
        divisor = np.array([lenOfCut]*200)
        return np.divide(vecSum, divisor)
        
if __name__ == "__main__":
    w2vs = W2VS()
    print(w2vs.getSenVec("今天天氣很好"))