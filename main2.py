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

def training_model(model, data, test_data, epoches, batches, training_status):
    print("Training started...")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()   # Multi-Class classification (x,y)  -> (LogSoftmax(x), One-Hot(y) OR y)
    encoder_input, decoder_input, decoder_output = data[0], data[1], data[2]
    test_encoder_input, test_decoder_input, test_decoder_output = test_data[0], test_data[1], test_data[2]

    for ep in range(1,epoches):
        encoder_inputI = iter(encoder_input)
        decoder_inputI = iter(decoder_input)
        decoder_outputI = iter(decoder_output)
        test_encoder_inputI = iter(test_encoder_input)
        test_decoder_inputI = iter(test_decoder_input)
        test_decoder_outputI = iter(test_decoder_output)
        for d in range(len(encoder_inputI)):
            model.train()
            encoder_input_ = next(encoder_inputI)
            decoder_input_ = next(decoder_inputI)
            decoder_output_ = next(decoder_outputI)
            #test_encoder_input_ = next(test_encoder_inputI)
            #test_decoder_input_ = next(test_decoder_inputI)
            #test_decoder_output_ = next(test_decoder_outputI)
            result = model(encoder_input_, decoder_input_, training_status)
            B, T, V = result.shape
            result = result.view(B*T,V)
            decoder_output_ = decoder_output_.view(B*T)
            loss = criterion(result, decoder_output_)
            opt.zero_grad()
            loss.backward()
            opt.step()

            result_test = None #testing_model(model, test_encoder_input_, test_decoder_input_, test_decoder_output_)
            
            #print(f"Results Translate: {[vocab[int(i)] for i in resultwords.tolist()]}")
            #print(f"Real (Decoder_Output): {[vocab[int(i)] for i in decoder_output_.tolist()]}")
            print('Train Epoches: {}/{} - Train Batches: {} - Train Loss: {} - Test Loss: {}'.format(ep, epoches, d, loss, result_test))

            if d % 20 == 0:
                torch.save(model.state_dict(), f"GEN_AI_weightsFull_{ep}_{d}.pt")

    return model


def testing_model(model, en_input, dec_input, dec_output):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        test_result = model(en_input,dec_input)
        B, T, V = test_result.shape
        test_result = test_result.view(B*T,V)
        dec_output = dec_output.view(B*T)
        loss = criterion(test_result, dec_output)
        return loss


def using_model(model, data_input, data_start):
    global decode
    encoder_input, decoder_input = data_input, data_start # Data, [START]
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    while True:
        with torch.no_grad():
            result = model(encoder_input,decoder_input, training_status)[:,-1,:]
            probs = softmax(result) # Probability for the Batch
            idx_next = torch.argmax(probs)  # Take token(s) with highest probability from Vocab
            decoder_input = torch.cat((decoder_input, idx_next), dim=1)[:,-max_token:] # Add token(s) to the sentence
            if (idx_next.tolist()[-1][-1] == 0) or (idx_next.tolist()[-1][-1] == 2):
                return True
            yield decoder_input

            
        


class TransformerModel(nn.Module):
    def __init__(self, vocab, max_token, num_blocks, num_heads, dim, in_features, out_features): #vocab = size of the vocab
        super().__init__()
        self.encoder = Encoder(vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        self.decoder = Decoder(vocab, max_token, num_blocks, num_heads, dim, dim, dim*2, dim)
        self.linear = nn.Linear(dim,in_features)
        self.linear2 = nn.Linear(in_features,out_features)
        self.activation = nn.ReLU()
        self.f_activation = nn.Softmax(dim=-1)

    def forward(self, input, input2, training_status):
        pad_mask_encoder = input != 0 # 2 == [PAD]
        pad_mask_encoder = pad_mask_encoder.unsqueeze(1).unsqueeze(1)
        pad_mask_decoder = input2 != 0 
        pad_mask_decoder = pad_mask_decoder.unsqueeze(1).unsqueeze(1)

        ahead_mask_decoder = input2 != 0
        mask = torch.triu(torch.ones(ahead_mask_decoder.shape[-1], ahead_mask_decoder.shape[-1])).transpose(0, 1).type(dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    
        encoder_output = self.encoder(input, pad_mask_encoder, training_status)
        decoder_output = self.decoder(input2, encoder_output, mask, pad_mask_decoder, training_status)   # Same mask for encoder and decoder [mask auestion pad mask for join, triangle for decoder at start]
        out = self.activation(self.linear(decoder_output))
        out = self.activation(self.linear2(out))#[:,-1,:]
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
    vocab.insert(0,"[END]") # 2
    vocab.insert(0,"[START]")  # 1
    vocab.insert(0,"[PAD]") # 0

    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


    sentE = []
    sentD = []
    sentDO = []
    for i in range(len(sentencesEncoder)):
        try:
            word = sentencesEncoder[i]  # Input Encoder
            num = encode(word)[:max_token]
            if len(num) < max_token:
                num += ([0] * (max_token - len(num)))  # Padding
            
            word2 = sentencesDecoder[i][7:] # Input Decoder
            num2 = encode(word2)
            num2.insert(0,1) #Insert [START]
            num2 = num2[:max_token]
            if len(num2) < max_token:
                num2 += ([0] * (max_token - len(num2)))   # Padding
            
            num3 = encode(word2)  # Output Decoder
            num3 = num3[:max_token-1]
            num3.append(2)  # Insert [END]
            if len(num3) < max_token:
                num3 += ([0] * (max_token - len(num3)))   # Padding

            sentE.append(num)
            sentD.append(num2)
            sentDO.append(num3)
        
        except:
            continue
        
    

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
    batches = 64      # Number of batches
    epoches = 10      # Number of epoches
    max_token = 256   # Amount of token per batch
    dim = 128         # Embedding dimension for a single token
    num_heads = 8     # Number of Heads per Attention Block
    num_blocks = 6    # Number of Attention Block

    train_encoder_input, train_decoder_input, train_decoder_output, test_encoder_input, test_decoder_input, test_decoder_output, len_vocab = get_data()              # Get the Data (images, labels)
    train_data = [train_encoder_input, train_decoder_input, train_decoder_output]
    test_data = [test_encoder_input, test_decoder_input, test_decoder_output]
    epoches, batches = epoches, batches
    model = TransformerModel(len_vocab, max_token, num_blocks, num_heads, dim, dim*num_heads*2, len_vocab)
    model.load_state_dict(torch.load("GEN_AI_weightsFull_1_0.pt"))
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    if True:
        training_status = True
        model = training_model(model,train_data,test_data,epoches,batches,training_status)
        torch.save(model.state_dict(), "GEN_AI_weights.pt")
        testing_model(model, test_data, epoches)
    
    if False:
        training_status = False
        model.load_state_dict(torch.load("GEN_AI_weightsFull_1_1360.pt"))
        while True:
            user_input = str(input(">"))
            user_input = torch.LongTensor([encode(user_input)])
            start_input = torch.LongTensor([[1]])
            for word in using_model(model, user_input, start_input):
                print("\033[H\033[2J", end="")
                #print(word.tolist()[-1])
                print(decode(word.tolist()[-1]))
    
    