"""BhavVaani: partial fine-tune of a Whisper encoder for Hindi speech emotion (Kaggle T4).

The bottom (L-K) encoder layers stay frozen and run once, exactly as in pre-training (30 s padding);
their output on the valid frames is cached. The top K layers + final LayerNorm + an attentive-statistics
pooling head are fine-tuned on the cached sequences (frame crop / time-mask augmentation).

CV: the 3 x 5 StratifiedKFold folds of the local pipeline (seeds 0, 1, 2). Outputs in /kaggle/working:
oof.npy [3, 826, 4], test.npy [209, 4] (mean of the 15 fold models), test_full.npy (all-train model),
bhav_whisper_ser/ (all-train weights + config for the Hugging Face release).
"""
import copy, glob, json, math, os, time
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import WhisperFeatureExtractor, WhisperModel

CFG = dict(backbone=os.environ.get('BACKBONE', 'openai/whisper-large-v3'), K=int(os.environ.get('K', 8)), epochs=12,
           bs=16, lr_layers=3e-5, lr_head=1e-3, wd=0.02, ls=0.1, seeds=[0, 1, 2], drop=0.3)
CLASSES = ['angry', 'happy', 'neutral', 'sad']
root = sorted(glob.glob('/kaggle/input/**/SRCASW-BhavVaani/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).values
paths = [f'{root}/train/{f}' for f in tr.filename] + [f'{root}/test/{f}' for f in te.filename]
dev = 'cuda'

fe = WhisperFeatureExtractor.from_pretrained(CFG['backbone'])
enc = WhisperModel.from_pretrained(CFG['backbone'], torch_dtype=torch.float16).encoder.to(dev).eval()
L = len(enc.layers); cut = L - CFG['K']
cache = []
t = time.time()
with torch.no_grad():
    for p in paths:
        w = sf.read(p, dtype='float32')[0][:30 * 16000]
        f = fe(w, sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
        hs = enc(f, output_hidden_states=True).hidden_states[cut]          # output of layer `cut`
        T = max(1, int(np.ceil(len(w) / 320)))
        cache.append(hs[0, :T].cpu())
print(f'cached layer {cut} of {L} in {time.time() - t:.0f}s', flush=True)
top_init = copy.deepcopy(nn.ModuleList(enc.layers[cut:])).float().cpu(); ln_init = copy.deepcopy(enc.layer_norm).float().cpu()
D = enc.config.d_model; del enc; torch.cuda.empty_cache()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = copy.deepcopy(top_init); self.ln = copy.deepcopy(ln_init)
        self.att = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(CFG['drop']), nn.Linear(2 * D, 256), nn.GELU(), nn.Dropout(CFG['drop']), nn.Linear(256, 4))

    def forward(self, x, mask):                                   # x B,T,D ; mask B,T
        am = (1.0 - mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        for lyr in self.layers:
            kw = {'layer_head_mask': None} if 'layer_head_mask' in lyr.forward.__code__.co_varnames else {}
            out = lyr(x, attention_mask=am, **kw)
            x = out[0] if isinstance(out, tuple) else out
        h = self.ln(x).float()
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.head(torch.cat([mu, sd], 1))


def batch(idx, train):
    xs = []
    for i in idx:
        x = cache[i]
        if train and len(x) > 50 and np.random.rand() < 0.5:
            n = np.random.randint(int(0.6 * len(x)), len(x) + 1); s = np.random.randint(0, len(x) - n + 1); x = x[s:s + n]
        xs.append(x)
    T = max(len(x) for x in xs)
    X = torch.zeros(len(xs), T, D, dtype=torch.float16); M = torch.zeros(len(xs), T)
    for j, x in enumerate(xs):
        X[j, :len(x)] = x; M[j, :len(x)] = 1
        if train and np.random.rand() < 0.5:
            n = len(x); wdt = np.random.randint(1, max(2, n // 5)); s = np.random.randint(0, max(1, n - wdt)); M[j, s:s + wdt] = 0
    return X.to(dev), M.to(dev)


def train(idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev)
    lay = list(net.layers.parameters()) + list(net.ln.parameters()); hd = list(net.att.parameters()) + list(net.head.parameters())
    opt = torch.optim.AdamW([{'params': lay, 'lr': CFG['lr_layers']}, {'params': hd, 'lr': CFG['lr_head']}], weight_decay=CFG['wd'])
    steps = CFG['epochs'] * math.ceil(len(idx) / CFG['bs'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [CFG['lr_layers'], CFG['lr_head']], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(CFG['epochs']):
        net.train(); perm = np.random.permutation(idx)
        for b in range(0, len(perm), CFG['bs']):
            bi = perm[b:b + CFG['bs']]; X, M = batch(bi, True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(net(X, M), torch.from_numpy(y[bi]).to(dev), label_smoothing=CFG['ls'])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    return net


@torch.no_grad()
def predict(net, idx):
    net.eval(); out = []
    for b in range(0, len(idx), 32):
        X, M = batch(idx[b:b + 32], False)
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(net(X, M).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


ntr = len(tr); te_idx = np.arange(ntr, len(paths))
oof = np.zeros((len(CFG['seeds']), ntr, 4)); test = np.zeros((len(te), 4))
for si, s in enumerate(CFG['seeds']):
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(ntr), y)):
        t = time.time(); net = train(a, 100 * s + k)
        oof[si, b] = predict(net, b); test += predict(net, te_idx) / (5 * len(CFG['seeds']))
        print(f'seed {s} fold {k} f1 {f1_score(y[b], oof[si, b].argmax(1), average="macro"):.4f} {time.time() - t:.0f}s', flush=True)
        del net; torch.cuda.empty_cache()
    print(f'seed {s} OOF macro-F1 {f1_score(y, oof[si].argmax(1), average="macro"):.4f}', flush=True)
np.save('/kaggle/working/oof.npy', oof); np.save('/kaggle/working/test.npy', test)
net = train(np.arange(ntr), 999); np.save('/kaggle/working/test_full.npy', predict(net, te_idx))
os.makedirs('/kaggle/working/bhav_whisper_ser', exist_ok=True)
torch.save({k: v.half() for k, v in net.state_dict().items()}, '/kaggle/working/bhav_whisper_ser/head_and_top_layers.pt')
json.dump(dict(CFG, classes=CLASSES, frozen_layers=cut, total_layers=L, d_model=D, sample_rate=16000),
          open('/kaggle/working/bhav_whisper_ser/config.json', 'w'), indent=1)
pd.DataFrame({'Id': te.Id}).to_csv('/kaggle/working/test_ids.csv', index=False)
print('done')
