# %%
from building_babel.tokenizers import LlamaTokenizer
from datasets import load_dataset
import torch
import building_babel.model as bbm
import torch.nn.functional as F
from logging import basicConfig, INFO
from torchtext.functional import to_tensor
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import CSVLogger

torch.set_float32_matmul_precision('medium')
# %%
basicConfig(level=INFO)
ds = load_dataset("roneneldan/TinyStories", split="train")
lt = LlamaTokenizer("/root/tokenizer.model")

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
config = bbm.TransformerConfig(128, 1, lt.n_words, head_dim=128, max_seq_len=5499)
sds = SimpleDataset(ds)
dl = torch.utils.data.DataLoader(sds, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=10)

class Babel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = bbm.Transformer(self.config)

    def training_step(self, batch, batch_idx):
        out = self.model(batch[..., :-1])
        loss = F.cross_entropy(out.transpose(1,2), batch[...,1:])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        return optimizer

b = Babel(config)
lg = CSVLogger("logs", name="testing")
ptp = L.pytorch.profilers.PyTorchProfiler(filename="./ptprofiler.txt", group_by_input_shapes=True, dirname="./logs/")
trainer = L.Trainer(limit_train_batches=100, max_epochs=1, profiler=ptp, callbacks=[DeviceStatsMonitor()], logger=lg)
trainer.fit(model=b, train_dataloaders=dl)

