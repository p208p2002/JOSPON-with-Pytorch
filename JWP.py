from gensim.models.doc2vec import Doc2Vec

class JWP():
    def __init__(self):
        self.d2vModel = Doc2Vec.load('d2vmodel/d2vmodel.model')