"""BhavVaani: attentive-pooling heads on frozen Whisper encoder frames for several backbones (Kaggle T4).

For each backbone: last encoder layer (after the final LayerNorm) on the valid frames, 30 s padded input as in
pre-training; then the head of bhav/scripts/attn_head.py trained with the 3 x 5 StratifiedKFold folds
(seeds 0, 1, 2), 30 epochs. Writes probs_attn_<name>.npz (oof[3,826,4], test[209,4]) and head_<name>_full.pt.
"""
import glob, math, os, time, traceback
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import WhisperFeatureExtractor, WhisperModel

CLASSES = ['angry', 'happy', 'neutral', 'sad']
root = sorted(glob.glob('/kaggle/input/**/SRCASW-BhavVaani/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).values
waves = [sf.read(f'{root}/train/{f}', dtype='float32')[0] for f in tr.filename] + \
        [sf.read(f'{root}/test/{f}', dtype='float32')[0] for f in te.filename]
ntr = len(tr); dev = 'cuda'


@torch.no_grad()
def frames(repo):
    fe = WhisperFeatureExtractor.from_pretrained(repo)
    enc = WhisperModel.from_pretrained(repo, torch_dtype=torch.float16).encoder.to(dev).eval()
    out = []
    for w in waves:
        f = fe(w[:30 * 16000], sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
        h = enc(f).last_hidden_state[0]
        out.append(h[:max(1, int(np.ceil(min(len(w), 480000) / 320)))].float().cpu())
    del enc; torch.cuda.empty_cache()
    return out


class Head(nn.Module):
    def __init__(self, D, H=256, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Dropout(p), nn.Linear(D, H), nn.GELU())
        self.att = nn.Sequential(nn.Linear(H, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * H, 4))

    def forward(self, x, mask):
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
    T = max(len(x) for x in xs); D = xs[0].shape[1]
    X = torch.zeros(len(xs), T, D); M = torch.zeros(len(xs), T)
    for j, x in enumerate(xs):
        X[j, :len(x)] = x; M[j, :len(x)] = 1
        if train and np.random.rand() < 0.5:
            L = len(x); w = np.random.randint(1, max(2, L // 5)); s = np.random.randint(0, max(1, L - w)); M[j, s:s + w] = 0
    return X.to(dev), M.to(dev)


def fit(seqs, yy, idx, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Head(seqs[0].shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
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


JOBS = [('whisper_large_v1', 'openai/whisper-large'), ('whisper_hi_medium', 'vasista22/whisper-hindi-medium'),
        ('whisper_large_v2', 'openai/whisper-large-v2'), ('whisper_hi_large_v2', 'vasista22/whisper-hindi-large-v2'),
        ('whisper_vaani_large_v3', 'ARTPARK-IISc/whisper-large-v3-vaani-hindi')]
for name, repo in JOBS:
    t = time.time()
    try:
        seqs = frames(repo); yy = np.r_[y, np.zeros(len(te), int)]
        oof = np.zeros((3, ntr, 4)); test = np.zeros((len(te), 4)); te_idx = np.arange(ntr, len(seqs))
        for si, s in enumerate([0, 1, 2]):
            for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(ntr), y)):
                net = fit(seqs, yy, a, 30, 100 * s + k)
                oof[si, b] = predict(net, seqs, b); test += predict(net, seqs, te_idx) / 15
            print(name, 'seed', s, 'OOF macro-F1', round(f1_score(y, oof[si].argmax(1), average='macro'), 4), flush=True)
        np.savez(f'/kaggle/working/probs_attn_{name}.npz', oof=oof, test=test)
        net = fit(seqs, yy, np.arange(ntr), 30, 999); torch.save(net.state_dict(), f'/kaggle/working/head_{name}_full.pt')
    except Exception:
        traceback.print_exc()
    print(f'{name} done in {time.time() - t:.0f}s', flush=True)
