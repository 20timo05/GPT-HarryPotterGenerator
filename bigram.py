import torch
import torch.nn.functional as F
import torch.nn as nn

# Create Vocabulary
with open('./data/HarryPotterPreprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

# Tokenization
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# split data into train & validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(data.shape[0] * 0.9)
train_data, val_data = data[:n], data[n:]

# Hyperparameters
VOCAB_SIZE = len(chars)
CONTEXT_SIZE = 8
BATCH_SIZE = 64
MAX_STEPS = 10000
EVAL_INTERVAL = 300
EVAL_LOSS_BATCHES = 200

# Loader that returns a batch
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - CONTEXT_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in ix])
    return x, y

# calculate mean loss for {EVAL_LOSS_BATCHES}x batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_LOSS_BATCHES)
        for i in range(EVAL_LOSS_BATCHES):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
    
    def forward(self, x, y=None):
        x = x.view(-1) # flatten => don't need the structure because of bigram model (x[0][0] should predict y[0][0])
        logits = self.token_embedding_table(x)
        
        if y is None:
            loss = None
        else:
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)

        return logits, loss
    
    def generate(self, first_char, max_new_tokens):
        output = first_char
        for _ in range(max_new_tokens):
            last_char = torch.tensor(stoi[output[-1]])
            
            # add batch dimension and feed to model
            logits, _ = self(last_char.view(1, -1))
            probs = F.softmax(logits, dim=-1)
            new_char = itos[torch.multinomial(probs, num_samples=1).item()]

            output += new_char

        return output


# Training Loop
model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(MAX_STEPS):
    # calculate loss every once in a while
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step}) train: {losses['train']:.4f}, val: {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Inference (Generate Harry Potter'ish text)
output = model.generate("\n", 500)
print(output)