import torch
import torch.nn as nn
import numpy as np
import math

from encoder import Encoder, MultiHeadAttention, EmbeddingAndPositional, AddAndNorm, FeedForwardNetwork


class MaskMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.scale = self.dim_per_head ** -0.5 
        self.k_ = nn.Linear(self.dim, self.dim)     # K matrix
        self.q_ = nn.Linear(self.dim, self.dim)     # Q matrix
        self.v_ = nn.Linear(self.dim, self.dim)     # V matrix
        self.f = nn.Linear(self.dim, self.dim)                        # Final FF

    def multiheaddot(self, K, Q, V, mask, training_status):
        out = ((Q @ K.permute(0,1,3,2)) / math.sqrt(Q.shape[-1])) * self.scale    # (Q * K) / sqrt(dim)
        if training_status:
            out = out.masked_fill(mask == 0, -1e9)
        out = torch.softmax(out,dim=-1)                                       # Apply softmax
        out = out @ V        
        return out

    def forward(self, query, key, value, pad_mask, training_status):
        query = self.q_(query) # (batch, numb_of_token, dim_per_token)
        query = query.view(query.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3) # (batch, numb_of_token, head, dim_per_head) -> (batch, head, numb_of_token, dim_per_head)
        key = self.k_(key)
        key = key.view(key.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3)
        value = self.v_(value)
        value = value.view(value.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3)


        out = self.multiheaddot(query, key, value, pad_mask, training_status)   # (batch, num_head, feat, dim_head)
        out = out.transpose(2,1) # (batch, feat, num_head, dim_head)
        out = out.flatten(2)
        out = self.f(out)
        
        
        return out


class BlockDecoder(nn.Module):
    def __init__(self, num_heads, dim, in_features, hidden_features, out_features):
        super().__init__()
        self.maskedmultihead = MaskMultiHeadAttention(num_heads, dim)
        self.multihead = MultiHeadAttention(num_heads, dim)
        self.addnorm_1 = AddAndNorm(dim)
        self.addnorm_2 = AddAndNorm(dim)
        self.addnorm_3 = AddAndNorm(dim)
        self.ffn = FeedForwardNetwork(in_features, hidden_features, out_features)

    def forward(self, input, input_encoder, input_mask, pad_mask, training_status):
        out = self.maskedmultihead(input, input, input, input_mask, training_status)
        out_ = self.addnorm_1(input,out)
        out = self.multihead(out_, input_encoder, input_encoder, pad_mask, training_status)
        out_ = self.addnorm_2(out_,out)
        out = self.ffn(out_)
        out = self.addnorm_3(out_,out)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab, max_token, num_blocks, num_heads, dim, in_features, hidden_features, out_features):
        super().__init__()
        self.embedpos = EmbeddingAndPositional(vocab, dim, max_token)
        self.blocks = nn.ModuleList([BlockDecoder(num_heads, dim, in_features, hidden_features, out_features) for _ in range(num_blocks)])

    def forward(self, input, encoder_input, input_mask, pad_mask, training_status):
        out = self.embedpos(input)
        
        for block in self.blocks:
            out = block(out, encoder_input, input_mask, pad_mask, training_status)

        return out
    

if __name__ == "__main__":

    if False: # Test Norm
        dataI = torch.randint(0,10,(10,5,10)).type(torch.float32)
        dataO = torch.randint(0,10,(10,5,10)).type(torch.float32)
        norm = AddAndNorm(10)
        out = norm(dataI,dataO)
        print(out)

    if True:
        max_token = 960
        dim = 256
        num_heads = 8
        num_blocks = 6

        vocab = ["[PAD]","[START]","[END]"]
        sentencesEncoder = ["This is fine", "I am good", "Hello"]
        for i in range(len(sentencesEncoder)):
            words = sentencesEncoder[i].split(" ")
            if len(words) < max_token:
                sentencesEncoder[i] += " [PAD]" * (max_token - len(words))
            vocab += words

        sentencesDecoder = ["Sure it is going to be", "Good, me as well", "Hello my dear friend!"]

        for i in range(len(sentencesDecoder)):
            words = sentencesDecoder[i].split(" ")
            words.insert(0,"[START]")
            words.append("[END]")
            sentencesDecoder[i] = "[START] " + sentencesDecoder[i] + " [END]"
            if len(words) < max_token:
                sentencesDecoder[i] += " [END]" * (max_token - len(words))
            vocab += words
    
    
        vocab = list(set(vocab))
        sentE = []
        sentD = []
        for i in sentencesEncoder:
            word = i.split(" ")
            num = [vocab.index(i) for i in word]
            sentE.append(num)
        
        for i in sentencesDecoder:
            word = i.split(" ")
            num = [vocab.index(i) for i in word]
            sentD.append(num)

        len_vocab = len(vocab)
        encoder = Encoder(len_vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        result_encoder = encoder(torch.LongTensor(sentE))
        print(result_encoder.shape)

        len_vocab = len(vocab)
        decoder = Decoder(len_vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        result = decoder(torch.LongTensor(sentD), result_encoder)
        print(result.shape)

    if False:
        embedding = EmbeddingAndPositional(len(vocab),dim,max_token)   # Vocab size, dim for one token, amount of tokens
        embeddresult = embedding(torch.LongTensor(sent))
        print(embeddresult.shape)

        block = BlockEncoder(num_heads,dim, dim, dim*2, dim)
        result = block(embeddresult)

        AttentionMulti = MultiHeadAttention(8,embeddresult.shape[-1])
        outAttention = AttentionMulti(embeddresult)
        print(outAttention.shape)





