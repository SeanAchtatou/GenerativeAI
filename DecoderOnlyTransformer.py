import torch
import torch.nn as nn
import numpy
import json
import csv
import math
import re

from torch.utils.data import DataLoader

epoches = 100
batches = 64
max_token = 2048
embedding = 128
block_nn_layer = 256
num_blocks = 8
num_heads = 8
dropout = 0.2

encode = None
decode = None
vocab = []


class PositionEmbedding(nn.Module):
    def __init__(self, num_token, size_embedding):
        super().__init__()
        self.pe = torch.zeros(num_token, size_embedding)
        self.pos = torch.arange(0,num_token,1).float().unsqueeze(1)
        self.embedding_index = torch.arange(0,size_embedding,2).float()
        self.denominator = 1 / torch.Tensor(10000 ** (self.embedding_index / size_embedding)) 
        
        self.pe[:, 0::2] = torch.sin(self.pos * self.denominator)
        self.pe[:, 1::2] = torch.cos(self.pos * self.denominator)
    
    def forward(self, x):
        out = x + self.pe[:x.shape[1], :]
        return out
    

class AddAndNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(self.dim)
    
    def forward(self, input_1, input_2):
        out = input_1 + input_2
        out = self.norm(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, dim_features):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim_features, 4 * dim_features),
                nn.ReLU(),
                nn.Linear(4 * dim_features, dim_features),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        out = self.net(x)
        return out
    

class Head(nn.Module):
    def __init__(self, dim, dim_per_head):
        super().__init__()
        self.K = nn.Linear(dim, dim_per_head)
        self.Q = nn.Linear(dim, dim_per_head)
        self.V = nn.Linear(dim, dim_per_head)
        self.scale = dim_per_head ** -0.5
        
    def forward(self, x, mask=None):
        key = self.K(x)
        query = self.Q(x)
        value = self.V(x)

        out = ((query @ key.permute(0,2,1)) / math.sqrt(query.shape[-1])) * self.scale 
        if mask:
            mask = torch.tril(torch.ones(x.shape[-2],x.shape[-2])).type(dtype=torch.uint8)
            out = out.masked_fill(mask == 0, -1e9)
        out = torch.softmax(out,dim=-1)                                       # Apply softmax
        out = out @ value                                                         # out * V
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dim_per_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(dim, dim_per_head) for _ in range(num_heads)])
    
    def forward(self, x):
        out = torch.cat([head(x,True) for head in self.heads], dim=-1)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = self.dim // self.num_heads

        self.attention = Attention(dim, self.num_heads, self.dim_per_head)
        self.addandnorm_1 = AddAndNorm(self.dim)
        self.addandnorm_2 = AddAndNorm(self.dim)
        self.ffnetwork = FeedForwardBlock(self.dim)
    
    def forward(self, input):
        out_attention = self.attention(input)
        out = self.addandnorm_1(out_attention, input)
        out_network = self.ffnetwork(out)
        out = self.addandnorm_2(out_network, out)
        return out


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocabulary, size_embedding, num_blocks, num_heads, nn_features, hidden_nn_features):
        super().__init__()
        self.num_blocks = num_blocks # Number of Blocks
        self.num_heads = num_heads # Number of Heads per block
        self.block_nn_features = nn_features # Amount of neurons for the neural network in each block
        self.hidden_nn_features = hidden_nn_features # Amount of neurons for the final neural network
        self.vocabulary = vocabulary # Vocabulary of all the sentences 
        self.size_embedding = size_embedding # Embedding of a single token

        self.embedding = nn.Embedding(self.vocabulary, self.size_embedding) # Embedding from Vocabulary to Embedding Size
        self.position_embedding = PositionEmbedding(max_token,size_embedding)
        self.blocks = nn.ModuleList([Block(self.size_embedding, self.num_heads) for _ in range(num_blocks)])
        #self.hidden_layer = nn.Linear(self.size_embedding, self.hidden_nn_features)
        self.out_layer = nn.Linear(self.size_embedding, self.vocabulary)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)

        for block in self.blocks:
            out = block(out)

        out = self.out_layer(out)
        return out

    


def training_model(model, data, test_data, epoches, batches):
    print("Training started...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()   # Multi-Class classification (x,y)  -> (LogSoftmax(x), One-Hot(y) OR y)
    decoder_input, decoder_output = data[0], data[1]
    #test_decoder_input, test_decoder_output = test_data[0], test_data[1]

    for ep in range(0,epoches):
        decoder_inputI = iter(decoder_input)
        decoder_outputI = iter(decoder_output)
        for d in range(0,len(decoder_inputI)):
            model.train()
            decoder_input_ = next(decoder_inputI)
            decoder_output_ = next(decoder_outputI)
        
            result = model(decoder_input_)
            B, T, V = result.shape
            result = result.view(B*T,V)
            decoder_output_ = decoder_output_.view(B*T)
            loss = criterion(result, decoder_output_)
            opt.zero_grad()
            loss.backward()
            opt.step()

            result_test = None #testing_model(model, test_encoder_input_, test_decoder_input_, test_decoder_output_)
            print('Train Epoches: {}/{} - Train Batches: {} - Train Loss: {} - Test Loss: {}'.format(ep, epoches, d, loss, result_test))

            #if d % 20 == 0:
             #   torch.save(model.state_dict(), f"GEN_AI_DecoderOnly_{ep}_{d}.pt")
    torch.save(model.state_dict(), f"GEN_AI_DecoderOnly_Final.pt")
    return model




def data():
    global vocab, encode, decode

    questions = []
    answers = []
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?!+-&@*ç%&/()=?`ü!£äö_:;éè¨$à-.,\"^'1234567890¦@#°§¬|¢´~][}{] "
    
    with open("data/data.json","r",encoding="utf-8") as f:
        dat = f.readlines()
        for line in dat:
            x = json.loads(line)
            question = re.sub(r"\\|'|/|\n",'', x["question"]).replace("  "," ")
            try:
                answer = re.sub(r"\\|'|/|\n",'', x["chatgpt_answers"][0]).replace("  "," ")
            except:
                answer = re.sub(r"\\|'|/|\n",'', x["human_answers"][0]).replace("  "," ")
            
            if all([w in list(set(alphabet)) for w in list(set(question))]) and all([w in list(set(alphabet)) for w in list(set(answer))]):
                questions.append(question)
                answers.append(answer)
                vocab += list(set(question + answer))
            


    vocab.append("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?!+-&@*ç%&/()=?`ü!£äö_:;éè¨$à-.,\"^'1234567890¦@#°§¬|¢´~][}{] ")
    vocab = sorted(list(set(" ".join(vocab))))
    vocab.insert(0,"[END]") # 2
    vocab.insert(0,"[START]")  # 1
    vocab.insert(0,"[PAD]") # 0

    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    encoded_samples = []
    encoded_labels = []
    for i in range(len(questions)):
        encoded_sample = encode(questions[i])
        encoded_label = encode(answers[i])
        final_input_encoded = encoded_sample + [1] + encoded_label + ([0]*(max_token - len(encoded_sample) - len(encoded_label) - 1))
        final_output_encoded = encoded_sample[1:] + [1] + encoded_label + [2] + ([0]*(max_token - len(encoded_sample[1:]) - len(encoded_label) - 2))
        encoded_samples.append(final_input_encoded[:max_token])
        encoded_labels.append(final_output_encoded[:max_token])

    ES = torch.LongTensor(encoded_samples)
    EL = torch.LongTensor(encoded_labels)
   
    shuffle = torch.randperm(len(ES))
    sentE, sentD = ES[shuffle], EL[shuffle]
    
    percentage = int((len(ES)/ 100) * 80)
    train_decoder_input, train_decoder_output = sentE[:percentage], sentD[:percentage]
    test_decoder_input, test_decoder_output = sentE[percentage:], sentD[percentage:]

    batch_size = batches
    train_decoder_input, train_decoder_output = DataLoader(train_decoder_input, batch_size=batch_size, shuffle=False), DataLoader(train_decoder_output, batch_size=batch_size, shuffle=False)
    test_decoder_input, test_decoder_output = DataLoader(test_decoder_input, batch_size=batch_size, shuffle=False), DataLoader(test_decoder_output, batch_size=batch_size, shuffle=False)
   
    
    return [train_decoder_input, train_decoder_output], [test_decoder_input, test_decoder_output]


def using_model(model, data_input):
    global decode
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    while True:
        with torch.no_grad():
            result = model(data_input)[:,-1,:]
            probs = softmax(result) # Probability for the Batch
            idx_next = torch.argmax(probs, dim=-1).unsqueeze(1)  # Take token(s) with highest probability from Vocab
            data_input = torch.cat((data_input, idx_next), dim=1)[:,-max_token:] # Add token(s) to the sentence
            if (idx_next[-1][-1] == 0) or (idx_next[-1][-1] == 2):
                return True
            yield data_input

# DATA > EMBEDDING > POSITIONAL EMBEDDING > BLOCK > ATTENTION > FFC
if __name__ == "__main__":
    

    train_data, test_data = data()

    model = DecoderOnlyTransformer(len(vocab), embedding, num_blocks, num_heads, block_nn_layer, len(vocab)*3)
    training_model(model, train_data, test_data, epoches, batches)

    if True:
        training_status = False
        #model.load_state_dict(torch.load("GEN_AI_DecoderOnly_Final.pt"))
        while True:
            user_input = str(input(">"))
            user_input = torch.LongTensor([encode(user_input)])
            start_input = torch.LongTensor([[1]])
            for word in using_model(model, user_input):
                print("\033[H\033[2J", end="")
                print(decode(word.tolist()[-1]))



# WE DO NO USE PADDING MASK IN A DECODER-ONLY TRANSFORMER

# IN ENCODER-DECODER TRANSFORMER - ENCODER PART WE USE PADDING MASK
#                                - DECODER PART WE USE THE TRIANGLE MASK- during self attention (first attention, not the combinaison of encoder and decoder) -> We dont use any other mask!
