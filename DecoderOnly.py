import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters
batch_size = 64 # Batches
block_size = 256 # Tokens
max_iters = 5000 # Epoches
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

stoi = None
itos = None
encode = None
decode = None
text = None
vocab_size = None

def data_():
    global stoi, itos, encode, decode, vocab_size, text
    with open('input.txt', 'r', encoding='utf-8') as f:  # Read the "input.txt" file to retrieve the data.
        text = f.read()

    chars = sorted(list(set(text))) # Amount of different characters found in the whole data
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) } # Dictionary for Encoding
    itos = { i:ch for i,ch in enumerate(chars) } # Dictionary for Decoding
    encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take a list of integers, output a string


def main():
    global stoi, itos, encode, decode, vocab_size, text
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # 90% of data for training
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):  # Randomly takes a set of data based on the block size and the batch size, and the corresponding labels
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])            # data
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])        # labels (switch by one on the right)
        x, y = x.to(device), y.to(device)
        return x, y

    """@torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out"""

    class Head(nn.Module): #AttentionHead in the AttentionBlock
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)  # K - From the embedding to the head size (defined as embedding by numbers of heads)
            self.query = nn.Linear(n_embd, head_size, bias=False) # Q - From the embedding to the head size (defined as embedding by numbers of heads)
            self.value = nn.Linear(n_embd, head_size, bias=False) # V - From the embedding to the head size (defined as embedding by numbers of heads)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #Mask
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B,T,C = x.shape
            k = self.key(x)   # (B, Token, Embedding) -> (B, Token, HeadSize)
            q = self.query(x) # (B, Token, Embedding) -> (B, Token, HeadSize)
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, Token, HeadSize) @ (B, HeadSize, Token) -> (B, Token, Token)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, Token, Token) with Mask applied
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei) 
            v = self.value(x) # (B, Token, Embedding) -> (B, Token, HeadSize)
            out = wei @ v # (B, Token, Token) @ (B, token, HeadSize) -> (B, Token, HeadSize)
            return out


    class MultiHeadAttention(nn.Module): #AttentionBlock in the Block
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Numbers of heads for the Attention block
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1) # Go through each Head Attention, and concatenate each Heads together (B, Token, HeadSize)* N = (B, Token, Embedding)
            out = self.dropout(self.proj(out))
            return out


    class FeedFoward(nn.Module): #FFN in the Block
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)


    class Block(nn.Module): #Block of the Encoder
        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x)) # MultiHeadAttention and Residual 
            x = x + self.ffwd(self.ln2(x)) #FFN and Residual
            return x


    class GPTLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Embedding for each vocabulary
            self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Embedding for the position of the vocabulary
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # Numbers of Attention Blocks
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            tok_emb = self.token_embedding_table(idx) # (B, Token) -> (B, Token, Embedding)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (Token) -> (Token, Embedding)
            x = tok_emb + pos_emb # (B, Token, Embedding)
            x = self.blocks(x) # (B, Token, Embedding)
            x = self.ln_f(x) # (B, Token, Embedding)
            logits = self.lm_head(x) # (B, Token, Vocab)

            if targets is None: #Testing phase?
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C) # (B, Token, Vocab) -> (B*Token, Vocab)
                targets = targets.view(B*T) # (Batch, Token) -> (Batch*Token)
                loss = F.cross_entropy(logits, targets) # Loss calculation

            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:] # Restrict to the max amount of Tokens (take the last amount of tokens)
                logits, loss = self(idx_cond) # Go trough the model
                logits = logits[:, -1, :] # (B, Token, Vocab) -> (B, Vocab)
                probs = F.softmax(logits, dim=-1) # Probability for each Batch
                idx_next = torch.multinomial(probs, num_samples=1) # Take token(s) with highest probability from Vocab
                idx = torch.cat((idx, idx_next), dim=1) # Add token(s) to the sentence(s)
                yield idx
            


    model = GPTLanguageModel()
    if False:
        model = GPTLanguageModel()
        m = model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        for i in range(max_iters): # Epoches
            m.train()
            xb, yb = get_batch('train') # Get Batch data for training
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            m.eval()
            xb, yb = get_batch('test') # Get Batch daza for testing
            logitsT, lossT = m(xb, yb)
            print(f"Training Loss : {loss} / Test Loss : {lossT}")

            if i % 100 == 0:
                torch.save(m.state_dict(), f"ENC_ONLY_AI_weights_{i}.pt")

        torch.save(m.state_dict(), "DEC_ONLY_AI_weights.pt")

    return model


def testModel(model):
    m = model.to(device)
    m.load_state_dict(torch.load("DEC_ONLY_AI_weights.pt"))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for i in m.generate(context, max_new_tokens=500):
        print(decode(i[0].tolist()))
        print("\033[H\033[2J", end="")
        

        
      


if __name__ == "__main__":
    data_()
    model_cb = main()
    testModel(model_cb)
