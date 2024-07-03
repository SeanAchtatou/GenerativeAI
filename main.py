import torch.nn as nn
import numpy as np
import torch
import csv

from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder


vocab = None
encode = None
decode = None

def training_model(model, data, test_data, epoches, batches):
    print("Training started...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()   # Multi-Class classification (x,y)  -> (LogSoftmax(x), One-Hot(y) OR y)
    encoder_input, decoder_input, decoder_output = data[0], data[1], data[2]

    model.train()
    for ep in range(epoches):
        encoder_inputI = iter(encoder_input)
        decoder_inputI = iter(decoder_input)
        decoder_outputI = iter(decoder_output)
        for d in range(0,len(encoder_input.dataset),batches):
            encoder_input_ = next(encoder_inputI)
            decoder_input_ = next(decoder_inputI)
            decoder_output_ = next(decoder_outputI)
            result = model(encoder_input_, decoder_input_)
            loss = criterion(result, decoder_output_)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            #print(f"Results Translate: {[vocab[int(i)] for i in resultwords.tolist()]}")
            #print(f"Real (Decoder_Output): {[vocab[int(i)] for i in decoder_output_.tolist()]}")
            print('Train Epoches: {}/{} -Train Batches: {} - Train Loss: {}'.format(ep, epoches, d, loss))

        if ep % 10 == 0:
                torch.save(model.state_dict(), f"GEN_AI_weightsFull_{ep}.pt")

    return model


def testing_model(model, data, epoches, batches):
    criterion = torch.nn.CrossEntropyLoss()

    encoder_input, decoder_input, decoder_output = data[0], data[1], data[2]
    model.eval()
    for epoches in range(epoches):
        with torch.no_grad():
            encoder_inputI = iter(encoder_input)
            decoder_inputI = iter(decoder_input)
            decoder_outputI = iter(decoder_output)
            for d in range(0,len(encoder_input.dataset),batches):
                encoder_input_ = next(encoder_inputI)
                decoder_input_ = next(decoder_inputI)
                decoder_output_ = next(decoder_outputI)

                result = model(encoder_input_,decoder_input_)
                loss = criterion(result, decoder_output_)
                print('Train Epoches: {}/{} -Train Batches: {} - Train Loss: {}'.format(epoches, epoches, d, loss))


def using_model(model, data):
    global decode
    encoder_input, decoder_input = data[0], data[1] # Data, [START]
    model.eval()
    while True:
        with torch.no_grad():
            result = model(encoder_input,decoder_input)
            probs = torch.nn.Softmax(result, dim=-1) # Probability for each Batch
            idx_next = torch.multinomial(probs, num_samples=1) # Take token(s) with highest probability from Vocab
            decoder_input = torch.cat((decoder_input, idx_next), dim=1) # Add token(s) to the sentence
            yield decoder_input

            if idx_next == 1:
                return True
        


class TransformerModel(nn.Module):
    def __init__(self, vocab, max_token, num_blocks, num_heads, dim, in_features, out_features): #vocab = size of the vocab
        super().__init__()
        self.encoder = Encoder(vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        self.decoder = Decoder(vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        self.linear = nn.Linear(dim,in_features)
        self.linear2 = nn.Linear(in_features,out_features)
        self.activation = nn.ReLU()
        self.f_activation = nn.Softmax(dim=-1)

    def forward(self, input, input2):
        pad_mask_encoder = input != 2 # 2 == [PAD]
        pad_mask_encoder = pad_mask_encoder.unsqueeze(1).unsqueeze(1)
        pad_mask_decoder = input2 != 2 
        pad_mask_decoder = pad_mask_decoder.unsqueeze(1).unsqueeze(1)

        ahead_mask_decoder = input2 != 2
        #ahead_mask_decoder = ahead_mask_encoder.unsqueeze(1)
        mask = torch.triu(torch.ones(ahead_mask_decoder.shape[-1], ahead_mask_decoder.shape[-1])).transpose(0, 1).type(dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
        #ahead_mask_encoder = ahead_mask_encoder & mask
        #ahead_mask_encoder = ahead_mask_encoder.unsqueeze(1)

    

        encoder_output = self.encoder(input, pad_mask_encoder)
        decoder_output = self.decoder(input2, encoder_output, pad_mask_decoder, mask)   # Same mask for encoder and decoder [mask auestion pad mask for join, triangle for decoder at start]
        #out = torch.flatten(decoder_output,-2,-1)
        out = self.activation(self.linear(decoder_output))
        out = self.activation(self.linear2(out))[:,-1,:]
        #out = self.f_activation(out)
        #out = torch.multinomial(out, num_samples=1)
        #out = out.view(-1)
        return out

            
def get_data():
    global vocab, encode, decode
    vocab = ["[PAD]","[START]","[END]"]

    sentencesEncoder_ = []
    sentencesDecoder_ = []
    
    opendata = open("data/data.csv", "r", newline="")
    readdata = csv.reader(opendata)

    for d in readdata:
        sentencesEncoder_.append(d[0].replace("  "," "))
        sentencesDecoder_.append(d[1].replace("  "," "))


      
    sentencesDecoder = []
    sentencesEncoder = []
    for i in range(len(sentencesDecoder_)):
        new_sentence_decoder_start = "[START]"
        sentencesDecoder.append(" ".join([new_sentence_decoder_start, sentencesDecoder_[i]]))
        sentencesEncoder.append(sentencesEncoder_[i])
        split_sent = set(sentencesDecoder_[i])
        vocab += split_sent

    vocab.append("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?!+*ç%&/()=?`ü!£äö_:;è¨$à-.,^'1234567890¦@#°§¬|¢´~][}{]")
    vocab = sorted(list(set(" ".join(vocab))))
    vocab.insert(0,"[PAD]") # 2
    vocab.insert(0,"[END]") # 1
    vocab.insert(0,"[START]")  # 0

    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


    sentE = []
    sentD = []
    sentDO = []
    for i in range(len(sentencesEncoder[:10000])):
        word = sentencesEncoder[i]
        num = encode(word)[:max_token]
        if len(num) < max_token:
            num += ([2] * (max_token - len(num))) 
        sentE.append(num)
        sentD.append([0] + ([2]* (max_token-1)))

        for j in range(len(sentencesDecoder[i][7:])):
            word2 = sentencesDecoder[i][7:8+j]
            num2 = encode(word2)
            sentDO.append(num2[-1])
            num2.insert(0,0)
            num2 = num2[:max_token]
            if len(num2) < max_token:
                num2 += ([2] * (max_token - len(num2))) 
            sentE.append(num)
            sentD.append(num2)
        sentDO.append(1)
    

    sentE = torch.LongTensor(sentE)
    sentD = torch.LongTensor(sentD)
    sentDO = torch.LongTensor(sentDO)

    shuffle = torch.randperm(len(sentE))
    sentE, sentD, sentDO = sentE[shuffle], sentD[shuffle], sentDO[shuffle]
    
    percentage = int((len(sentE)/ 100) * 80)
    train_encoder_input, train_decoder_input, train_decoder_output = sentE[:percentage], sentD[:percentage], sentDO[:percentage]
    test_encoder_input, test_decoder_input, test_decoder_output = sentE[percentage:], sentD[percentage:], sentDO[percentage:]
    
    batch_size = batches
    train_encoder_input, train_decoder_input, train_decoder_output = DataLoader(train_encoder_input, batch_size=batch_size, shuffle=False), DataLoader(train_decoder_input, batch_size=batch_size, shuffle=False), DataLoader(train_decoder_output, batch_size=batch_size, shuffle=False)
    test_encoder_input, test_decoder_input, test_decoder_output = DataLoader(test_encoder_input, batch_size=batch_size, shuffle=False), DataLoader(test_decoder_input, batch_size=batch_size, shuffle=False), DataLoader(test_decoder_output, batch_size=batch_size, shuffle=False)
   

    return train_encoder_input, train_decoder_input, train_decoder_output, test_encoder_input, test_decoder_input, test_decoder_output, len(vocab)
    


    


if __name__ == "__main__":
    batches = 256
    epoches = 100
    max_token = 256
    dim = 32
    num_heads = max_token // dim
    num_blocks = 6


    train_encoder_input, train_decoder_input, train_decoder_output, test_encoder_input, test_decoder_input, test_decoder_output, len_vocab = get_data()              # Get the Data (images, labels)
    train_data = [train_encoder_input, train_decoder_input, train_decoder_output]
    test_data = [test_encoder_input, test_decoder_input, test_decoder_output]
    epoches, batches = epoches, batches
    model = TransformerModel(len_vocab, max_token, num_blocks, num_heads, dim, dim*num_heads*2, len_vocab)

    if True:
        model = training_model(model,train_data,test_data,epoches,batches)
        torch.save(model.state_dict(), "GEN_AI_weights.pt")
        testing_model(model, test_data, epoches)
    
    if True:
        #model.load_state_dict(torch.load("GEN_AI_weights.pt"))
        while True:
            user_input = str(input(">"))
            user_input = torch.Tensor(encode([user_input]))
            start_input = torch.Tensor(encode(["[START]"]))
            for word in using_model(model, user_input):
                print(decode(word))
    
    