# -*- coding: UTF-8 -*-
"""
轉換資料集到句子向量
save as pickle
"""
from w2v_sentence_vector import W2VS
import pickle
import csv

if __name__ == "__main__":
    w2vs = W2VS()
    sentencesDict = {}
    with open('dataset/waimai_10k_tw.csv',newline='') as f:
        rows = csv.reader(f)
        for i,row in enumerate(rows):
            if(row[0] == 'label'):
                continue
            line = row[1].strip('\n')
            sVec = w2vs.getSenVec(line)
            # sentencesDict.append(sVec)
            sentencesDict[str(i-1)] = (sVec,row[0])
    with open('dataset/waimai_10k_tw.pkl','wb') as f:
        pickle.dump(sentencesDict,f)
    print("finish")
    