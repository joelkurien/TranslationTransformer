from datasets import load_dataset
from DataEncoding import DataEncoding
from torch import nn
import time
from Operations import Operations
from Encoder import Encoder
from Decoder import Decoder
from .AdamOptimizer import Adam

class Transformer:
    def __init__(self, encoder_layers, decoder_layers):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_weights = [[0,0,0,0]]*self.encoder_layers
        self.decoder_mask_weights = [[0,0,0,0]]*self.decoder_layers
        self.decoder_weights = [[0,0,0,0]]*self.decoder_layers
        self.operation = Operations()
    
    def forward(self, eng_embedded, fr_embedded):
        encoder = Encoder()
        decoder = Decoder()
        
        for i in range(self.encoder_layers):
            Wqe, Wke, Wve, Woe = self.operation.generate_self_weights_QKV(eng_embedded)
            self.encoder_weights[i] = [Wqe, Wke, Wve, Woe]
                    
        for i in range(self.decoder_layers):
            Wqmd, Wkmd, Wvmd, Womd = self.operation.generate_self_weights_QKV(fr_embedded)
            self.decoder_mask_weights[i] = [Wqmd, Wkmd, Wvmd, Womd]
            
            Wqd, Wkd, Wvd, Wod = self.operation.generate_cross_weights_QKV(fr_embedded, eng_embedded)
            self.decoder_weights[i] = [Wqd, Wkd, Wvd, Wod]
        
        outpt_probs = None  
        for layer in range(self.encoder_layers):
                eng_embedded = encoder.forward(eng_embedded, self.encoder_weights[layer])
            
        for _ in range(self.decoder_layers):
            fr_embedded = decoder.forward(fr_embedded, eng_embedded, self.decoder_weights[layer], self.decoder_mask_weights[layer])
    
        batch_size, seq_len, d_fr = fr_embedded.size()
        linr = nn.Linear(in_features = d_fr, out_features=d_fr)
        fr_embedded = linr(fr_embedded)
        fr_embedded = fr_embedded - fr_embedded.max(dim=-1, keepdim=True).values

        softmx = nn.Softmax(dim=-1)
        return softmx(fr_embedded)
    
    def train(self, eng_embedded, fr_embedded, epochs=None):
        for epoch in range(10):
            outpt_probs = self.forward(eng_embedded, fr_embedded)
            
        return outpt_probs
        
                
ds = load_dataset("wmt/wmt14", "fr-en", split="train", num_proc=4)

translation = ds["translation"][:3]
english = []
french = []
for sentences in translation:
    english.append(sentences["en"])
    french.append(sentences["fr"])
eng_word2Vec = DataEncoding(english)
eng_embedded = eng_word2Vec.input_data()
fr_word2Vec = DataEncoding(french)
fr_embedded = fr_word2Vec.input_data()

transformer = Transformer(encoder_layers=6, decoder_layers=6)
outpt_probs = transformer.train(eng_embedded, fr_embedded)
print(outpt_probs)