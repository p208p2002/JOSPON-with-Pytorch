from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
import jieba

# jieba初始化
jieba.set_dictionary('dict/dict.txt.big')
jieba.load_userdict('dict/my_dict')
jieba.initialize()

# 讀入waimai_10k_tw.csv，並且使用jieba斷詞
sentences = []
with open('dataset/waimai_10k_tw.csv',newline='') as f:
    rows = csv.reader(f)
    for row in rows:
        if(row[0] == 'label'):
            continue
        line = row[1].strip('\n')
        sentence = jieba.cut(line, cut_all=False)
        sentence = list(sentence)
        sentences.append(sentence)

# 資料準備
tagged_data = [TaggedDocument(sentence, [str(i)]) for i, sentence in enumerate(sentences)]

# train
max_epochs = 50
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025,min_count=1, dm =1)
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save('d2vmodel/d2vmodel.model')
print('finish')