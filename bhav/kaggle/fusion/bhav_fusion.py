"""BhavVaani fusion head (Kaggle T4): one attentive-pooling head over the frame-wise concatenation of the
last encoder layers of several Whisper backbones (all run at 50 frames/s, so frames align). Seeds A and B per fold.
Writes fusion.npz with oof_a/test_a/oof_b/test_b.
"""
import glob, math, os, time, traceback
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import WhisperFeatureExtractor, WhisperModel

BACKBONES = ['openai/whisper-large-v2', 'openai/whisper-large', 'ARTPARK-IISc/whisper-large-v3-vaani-hindi', 'vasista22/whisper-hindi-medium']
CLASSES = ['angry', 'happy', 'neutral', 'sad']
root = sorted(glob.glob('/kaggle/input/**/SRCASW-BhavVaani/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).values
waves = [sf.read(f'{root}/train/{f}', dtype='float32')[0] for f in tr.filename] + \
        [sf.read(f'{root}/test/{f}', dtype='float32')[0] for f in te.filename]
ntr = len(tr); dev = 'cuda'; te_idx = np.arange(ntr, len(waves))
yy = np.r_[y, np.zeros(len(te), int)]



@torch.no_grad()
def last_frames(repo):
    fe = WhisperFeatureExtractor.from_pretrained(repo)
    enc = WhisperModel.from_pretrained(repo, torch_dtype=torch.float16).encoder.to(dev).eval()
    out = []
    for w in waves:
        w = w[:29 * 16000]; T = max(1, int(np.ceil(len(w) / 320)))
        f = fe(w, sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
        out.append(enc(f).last_hidden_state[0, :T].cpu())
    del enc; torch.cuda.empty_cache()
    return out


class Head(nn.Module):
    def __init__(self, D, nl=1, H=256, p=0.3):
        super().__init__()
        self.nl = nl
        if nl > 1:
            self.lnl = nn.LayerNorm(D, elementwise_affine=False); self.lw = nn.Parameter(torch.zeros(nl))
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Dropout(p), nn.Linear(D, H), nn.GELU())
        self.att = nn.Sequential(nn.Linear(H, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * H, 4))

    def forward(self, x, mask):                     # x B,T,D or B,T,L,D
        if self.nl > 1:
            x = (F.softmax(self.lw, 0)[None, None, :, None] * self.lnl(x)).sum(2)
        h = self.inp(x)
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.out(torch.cat([mu, sd], 1))


def batch(seqs, idx, train, maxT=400):
    xs = []
    for i in idx:
        x = seqs[i]
        if train and len(x) > 50 and np.random.rand() < 0.5:
            L = np.random.randint(int(0.6 * len(x)), len(x) + 1); s = np.random.randint(0, len(x) - L + 1); x = x[s:s + L]
        xs.append(x[:maxT])
    T = max(len(x) for x in xs)
    X = torch.zeros((len(xs), T) + tuple(xs[0].shape[1:])); M = torch.zeros(len(xs), T)
    for j, x in enumerate(xs):
        X[j, :len(x)] = x.float(); M[j, :len(x)] = 1
        if train and np.random.rand() < 0.5:
            L = len(x); w = np.random.randint(1, max(2, L // 5)); s = np.random.randint(0, max(1, L - w)); M[j, s:s + w] = 0
    return X.to(dev), M.to(dev)


def fit(seqs, idx, seed, nl, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    D = seqs[0].shape[-1]; net = Head(D, nl).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx) / 32); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for b in range(0, len(perm), 32):
            bi = perm[b:b + 32]; X, M = batch(seqs, bi, True)
            loss = F.cross_entropy(net(X, M), torch.from_numpy(yy[bi]).to(dev), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


@torch.no_grad()
def predict(net, seqs, idx):
    net.eval(); out = []
    for b in range(0, len(idx), 64):
        X, M = batch(seqs, idx[b:b + 64], False); out.append(net(X, M).softmax(1).cpu().numpy())
    return np.concatenate(out)


t = time.time()
parts = []
for repo in BACKBONES:
    parts.append(last_frames(repo)); print(repo, 'extracted', round(time.time() - t), flush=True)
fused = [torch.cat([p[i] for p in parts], 1) for i in range(len(waves))]
del parts
print('fused dim', fused[0].shape, flush=True)
R = {k: np.zeros((3, ntr, 4)) for k in ['oof_a', 'oof_b']}
R.update({k: np.zeros((len(te), 4)) for k in ['test_a', 'test_b']})
for si, s in enumerate([0, 1, 2]):
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(ntr), y)):
        for tag, off in [('a', 0), ('b', 5000)]:
            net = fit(fused, a, off + 100 * s + k, 1)
            R[f'oof_{tag}'][si, b] = predict(net, fused, b); R[f'test_{tag}'] += predict(net, fused, te_idx) / 15
    for tag in 'ab':
        print('seed', s, tag, round(f1_score(y, R[f'oof_{tag}'][si].argmax(1), average='macro'), 4), flush=True)
np.savez('/kaggle/working/fusion.npz', **R)
print('done', round(time.time() - t), flush=True)
