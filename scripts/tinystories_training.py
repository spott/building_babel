# %%
from building_babel.tokenizers import LlamaTokenizer
from datasets import load_dataset
import torch
from tqdm.autonotebook import tqdm
import building_babel.model as bbm
import torch.nn.functional as F
from logging import basicConfig, INFO
from torchtext.functional import to_tensor

# %%
basicConfig(level=INFO)
ds = load_dataset("roneneldan/TinyStories", split="train")
lt = LlamaTokenizer("/Users/spott/Models/llama-2-tokenizer/tokenizer.model")

# %%
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len = 1024):
        self.df = df['text']
        self.max_len = max_len #df['text'].apply(len).max()

    def __getitem__(self, i):
        tokenized = lt.encode(self.df[i], True, True)
        #pad = (0,max(0,self.max_len - len(tokenized)))
        #return F.pad(tokenized, pad, "constant", 0) # we pad with 0s, but it really doesn't matter, because we have a stop token...
        return tokenized

    def __len__(self):
        return len(self.df)

# %%
def sample(x):
    return torch.multinomial(x, 1)

# %%
def generate(t, deterministic=False):
    seq = torch.tensor([[lt.bos_id]])
    for _ in range(18):
        if deterministic:
            next_token = t(seq)[:,-1].softmax(dim=-1).argmax(dim=-1).view(-1,1)
        else:
            next_token = sample(t(seq)[:,-1].softmax(dim=-1)).view(-1,1)
        if next_token[0,-1] < 2:
            break
        seq = torch.concat([seq, next_token], dim=-1)
    print(lt.decode(seq[:,1:].tolist()))

def collate_fn(ts):
    return to_tensor(ts, padding_value=0)


# %%
sds = SimpleDataset(ds)
c = bbm.TransformerConfig(128, 1, lt.n_words, head_dim=128, max_seq_len=5499)
t = bbm.Transformer(c)
dl = torch.utils.data.DataLoader(sds, batch_size=10, shuffle=True, collate_fn=collate_fn)

# %%
optim = torch.optim.Adam(t.parameters(), lr=3e-5)

# %%
for i in range(20):
    print(i)
    for b in tqdm(dl):
        optim.zero_grad()
        out = t(b[...,:-1])

        loss = F.cross_entropy(out.transpose(1,2), b[...,1:])

        loss.backward()
        optim.step()
    with torch.no_grad():
        generate(t, deterministic=True)

# %%
with torch.no_grad():
        generate(t, deterministic=True)

# %%



