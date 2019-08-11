# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("wikidata/wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=200)
    model.save("w2vmodel/word2vec.model")

if __name__ == "__main__":
    main()