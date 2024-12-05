from sklearn.feature_extraction.text import TfidfVectorizer
import torch 
from torch import nn, Tensor, inf

class DataEncoding:
    def __init__(self, corpus):
        self.corpus = corpus
    
    def __tfidf_matrix_generator(self):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.corpus)
    
    def __positional_encoding(self):
        tfidf = self.__tfidf_matrix_generator()
        seq_length, no_terms = tfidf.shape
        pos_vals = torch.arange(seq_length).unsqueeze(0)
        
        positional_encoder = nn.Embedding(seq_length, 512)
        return positional_encoder(pos_vals)
    
    def input_data(self):
        tfidf_dense = torch.tensor(self.__tfidf_matrix_generator().toarray(), dtype=torch.float32)
        return tfidf_dense + self.__positional_encoding()