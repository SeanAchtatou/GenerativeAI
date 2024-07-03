import torch
import torch.nn as nn
import numpy as np
import math



class PositionalEncoding():
    def __init__(self, dim, max_token, n=10_000):
        self.dim = dim
        self.max_token = max_token
        self.n = n
        
    def run(self):
        position_matrix = np.zeros((self.max_token,self.dim))   # Amount of token, and dimension per token
        for i in range(self.max_token):                    # For each token
            for j in np.arange(int(self.dim/2)):       # Each dimension of the token
                denominator = np.power(self.n, 2*j/self.dim)
                position_matrix[i, 2*j] = np.sin(i / denominator)
                position_matrix[i, (2*j) + 1] = np.cos(i / denominator)
  
        return torch.Tensor(position_matrix).unsqueeze(0)


class EmbeddingAndPositional(nn.Module):
    def __init__(self, vocab_size, dim, max_token):  #dim_ = max_token
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)    # Vocab_size and out dimension
        self.position = nn.Embedding(max_token, dim)
        self.positional = PositionalEncoding(dim, max_token)   # Calculate position for each token
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.embedding(input)
        out = out + self.position(torch.arange(out.shape[-2]))
        #out = out + self.positional.run()              # Sum of embedding and position
        out = self.dropout(out)
        return out
    

class AddAndNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layerNorm = nn.LayerNorm(self.dim)    # Dimensionality of the Feature layer

    def forward(self, output, input):
        out = input + output                       # Add
        out = self.layerNorm(out)                  # Norm
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_per_head = dim // num_heads
        self.scale = self.dim_per_head ** -0.5 
        #self.k_ = nn.Linear(self.dim_per_head, self.dim_per_head)     # K matrix
        #self.q_ = nn.Linear(self.dim_per_head, self.dim_per_head)     # Q matrix
        #self.v_ = nn.Linear(self.dim_per_head, self.dim_per_head)     # V matrix
        self.k_ = nn.Linear(self.dim, self.dim)     # K matrix
        self.q_ = nn.Linear(self.dim, self.dim)     # Q matrix
        self.v_ = nn.Linear(self.dim, self.dim)     # V matrix
        self.f = nn.Linear(self.dim, self.dim)                        # Final FF

    def multiheaddot(self, Q, K, V, mask, training_status):
        out = ((Q @ K.permute(0,1,3,2)) / math.sqrt(Q.shape[-1])) * self.scale    # (Q * K) / sqrt(dim)
        #mask = mask[:,None,None,:].expand(out.shape[0],out.shape[1],mask.shape[-1],mask.shape[-1])
        if training_status:
            out = out.masked_fill(mask == 0, -1e9)
        out = torch.softmax(out,dim=-1)                                       # Apply softmax
        out = out @ V                                                         # out * V

        return out

    def forward(self, query, key, value, pad_mask, training_status):
        query = self.q_(query) # (batch, numb_of_token, dim_per_token)
        query = query.view(query.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3) # (batch, numb_of_token, dim_per_token) -> (batch, numb_of_token, head, dim_per_head) -> (batch, head, numb_of_token, dim_per_head)
        key = self.k_(key)
        key = key.view(key.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3)
        value = self.v_(value)
        value = value.view(value.shape[0], -1, self.num_heads, self.dim_per_head).permute(0,2,1,3)


        out = self.multiheaddot(query, key, value, pad_mask, training_status)   # (batch, num_head, feat, dim_head)
        out = out.transpose(2,1) # (batch, feat, num_head, dim_head)
        out = out.flatten(2)
        out = self.f(out)
        
        return out
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.linear_1 = nn.Linear(self.in_features, self.hidden_features)
        self.linear_2 = nn.Linear(self.hidden_features, self.out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.act(self.linear_1(input))
        out = self.linear_2(self.dropout(out))
        return out
    

class BlockEncoder(nn.Module):
    def __init__(self, num_heads, dim, in_features, hidden_features, out_features):
        super().__init__()
        self.multihead = MultiHeadAttention(num_heads, dim)
        self.addnorm_1 = AddAndNorm(dim)
        self.addnorm_2 = AddAndNorm(dim)
        self.ffn = FeedForwardNetwork(in_features, hidden_features, out_features)

    def forward(self, input, pad_mask, training_status):
        out = self.multihead(input, input, input, pad_mask, training_status)
        out_ = self.addnorm_1(input,out)
        out = self.ffn(out_)
        out = self.addnorm_2(out_,out)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab, max_token, num_blocks, num_heads, dim, in_features, hidden_features, out_features):
        super().__init__()
        self.embedpos = EmbeddingAndPositional(vocab, dim, max_token)
        self.blocks = nn.ModuleList([BlockEncoder(num_heads, dim, in_features, hidden_features, out_features) for _ in range(num_blocks)])

    def forward(self, input, pad_mask, training_status):
        out = self.embedpos(input)
        
        for block in self.blocks:
            out = block(out, pad_mask, training_status)

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
        sentences = ["This is fine", "I am good", "Hello"]

        for i in range(len(sentences)):
            words = sentences[i].split(" ")
            if len(words) < max_token:
                sentences[i] += " [PAD]" * (max_token - len(words))
            vocab += words
    
        vocab = list(set(vocab))
        sent = []
        for i in sentences:
            word = i.split(" ")
            num = [vocab.index(i) for i in word]
            sent.append(num)

        len_vocab = len(vocab)
        encoder = Encoder(len_vocab, max_token, num_blocks, num_heads, dim, dim, dim*4, dim)  # len_vocab = vocabulary sizes (all words) -  max_token = maximum of token as input - dim = dimension per token
        result = encoder(torch.LongTensor(sent))
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





