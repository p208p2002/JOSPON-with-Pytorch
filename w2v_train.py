# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("wikidata/wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=3)

    #保存模型，供日後使用
    model.save("w2vmodel/word2vec2.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()