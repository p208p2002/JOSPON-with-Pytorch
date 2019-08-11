import jieba
from W2V_SV import W2VS
jieba.set_dictionary('dict/dict.txt.big')
jieba.load_userdict('dict/my_dict')
jieba.initialize()
w2vs = W2VS()
while True:
    s = input("測試句/詞:")
    l = list(jieba.cut(s))
    print(l)
    print(w2vs.getSenVec(s))
