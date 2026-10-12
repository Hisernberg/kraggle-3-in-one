"""BhavVaani bagging round (Kaggle T4): for each backbone, last-layer attentive-pooling heads with a third seed
('sl3') and a self-trained variant ('st': the fold model pseudo-labels its validation fold + test clips with max
probability > ST, then a new head is trained on the training fold plus those pseudo-labels; no true validation labels
are used). Same 3 x 5 folds as the local pipeline. Writes bag_<name>.npz with oof_sl3/test_sl3/oof_st/test_st.
"""
import glob, math, os, time, traceback
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import WhisperFeatureExtractor, WhisperModel

JOBS = [('whisper_large_v2', 'openai/whisper-large-v2'), ('whisper_vaani_large_v3', 'ARTPARK-IISc/whisper-large-v3-vaani-hindi')]
ST = 0.9
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


def fit(seqs, idx, seed, nl, epochs=30, labels=None):
    lab = yy if labels is None else labels
    torch.manual_seed(seed); np.random.seed(seed)
    D = seqs[0].shape[-1]; net = Head(D, nl).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx) / 32); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for b in range(0, len(perm), 32):
            bi = perm[b:b + 32]; X, M = batch(seqs, bi, True)
            loss = F.cross_entropy(net(X, M), torch.from_numpy(lab[bi]).to(dev), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


@torch.no_grad()
def predict(net, seqs, idx):
    net.eval(); out = []
    for b in range(0, len(idx), 64):
        X, M = batch(seqs, idx[b:b + 64], False); out.append(net(X, M).softmax(1).cpu().numpy())
    return np.concatenate(out)


for name, repo in JOBS:
    t = time.time()
    try:
        last = last_frames(repo); print(name, 'extracted', round(time.time() - t), flush=True)
        R = {k: np.zeros((3, ntr, 4)) for k in ['oof_sl3', 'oof_st']}
        R.update({k: np.zeros((len(te), 4)) for k in ['test_sl3', 'test_st']})
        for si, s in enumerate([0, 1, 2]):
            for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(ntr), y)):
                net = fit(last, a, 13000 + 100 * s + k, 1)
                pv = predict(net, last, b); pt = predict(net, last, te_idx)
                R['oof_sl3'][si, b] = pv; R['test_sl3'] += pt / 15
                U = np.r_[b, te_idx]; pu = np.r_[pv, pt]; keep = pu.max(1) > ST
                lab = yy.copy(); lab[U[keep]] = pu.argmax(1)[keep]
                net = fit(last, np.r_[a, U[keep]], 17000 + 100 * s + k, 1, labels=lab)
                R['oof_st'][si, b] = predict(net, last, b); R['test_st'] += predict(net, last, te_idx) / 15
            for key in ['oof_sl3', 'oof_st']:
                print(name, 'seed', s, key, round(f1_score(y, R[key][si].argmax(1), average='macro'), 4), flush=True)
        np.savez(f'/kaggle/working/bag_{name}.npz', **R)
    except Exception:
        traceback.print_exc()
    print(f'{name} done in {time.time() - t:.0f}s', flush=True)
