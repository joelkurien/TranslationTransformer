from sklearn.feature_extraction.text import TfidfVectorizer
import torch 
from gensim.models import Word2Vec
from torch import nn, Tensor, inf

class DataEncoding:
    def __init__(self, corpus):
        self.corpus = corpus
    
    def __word2vec_generator(self):
        input_embed_model = Word2Vec(self.corpus, vector_size=512, window=5, min_count =1, workers = 4)
        vocab = list(input_embed_model.wv.index_to_key)
        return input_embed_model.wv[vocab]
    
    def __positional_encoding(self):
        word2vec_mat = self.__word2vec_generator()
        seq_length, no_terms = word2vec_mat.shape
        pos_vals = torch.arange(seq_length).unsqueeze(0)
        
        positional_encoder = nn.Embedding(seq_length, 512)
        return positional_encoder(pos_vals)
    
    def input_data(self):
        tfidf_dense = torch.tensor(self.__word2vec_generator(), dtype=torch.float32)
        return tfidf_dense + self.__positional_encoding()