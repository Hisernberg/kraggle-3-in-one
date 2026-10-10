"""Attentive-statistics-pooling head over frozen Whisper encoder frames (CPU training).

python attn_head.py <frames npz name> <out name> [epochs]
OOF uses the same 3 x 5 folds as blend.py; the test prediction is the mean of the 15 fold models.
Writes work/probs/<out>.npz (oof[3,N,4], test[Ntest,4]) and work/models/<out>_full.pt (all-train model).
"""
import sys, os, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend

torch.set_num_threads(4)


class Head(nn.Module):
    def __init__(self, D, H=256, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Dropout(p), nn.Linear(D, H), nn.GELU())
        self.att = nn.Sequential(nn.Linear(H, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * H, 4))

    def forward(self, x, mask):                    # x B,T,D  mask B,T (1 = valid)
        h = self.inp(x)
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-5).sqrt()
        return self.out(torch.cat([mu, sd], 1))


def load_frames(name, m):
    z = np.load(f'{EMB}/frames_{name}.npz')
    key = dict(zip(zip(z['split'], z['ids']), range(len(z['ids']))))
    off = z['offsets']; fr = z['frames']
    return [fr[off[key[(s, i)]]:off[key[(s, i)] + 1]] for s, i in zip(m.split, m.Id)]


def batch(seqs, idx, train, maxT=400):
    xs = []
    for i in idx:
        x = seqs[i]
        if train and len(x) > 50 and np.random.rand() < 0.5:     # random crop, at least 60 % of the clip
            L = np.random.randint(int(0.6 * len(x)), len(x) + 1); s = np.random.randint(0, len(x) - L + 1); x = x[s:s + L]
        xs.append(x[:maxT])
    T = max(len(x) for x in xs); D = xs[0].shape[1]
    X = np.zeros((len(xs), T, D), np.float32); M = np.zeros((len(xs), T), np.float32)
    for j, x in enumerate(xs):
        X[j, :len(x)] = x; M[j, :len(x)] = 1
    if train:                                                    # time masking
        for j in range(len(xs)):
            if np.random.rand() < 0.5:
                L = int(M[j].sum()); w = np.random.randint(1, max(2, L // 5)); s = np.random.randint(0, max(1, L - w))
                M[j, s:s + w] = 0
    return torch.from_numpy(X), torch.from_numpy(M)


def fit(seqs, y, idx, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Head(seqs[0].shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx) / 32); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx)
        for b in range(0, len(perm), 32):
            bi = perm[b:b + 32]; X, M = batch(seqs, bi, True)
            loss = F.cross_entropy(net(X, M), torch.from_numpy(y[bi]), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net


@torch.no_grad()
def predict(net, seqs, idx):
    net.eval(); out = []
    for b in range(0, len(idx), 64):
        X, M = batch(seqs, idx[b:b + 64], False); out.append(net(X, M).softmax(1).numpy())
    return np.concatenate(out)


if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]; epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    m = load_meta(); seqs = load_frames(name, m)
    tr = np.where(m.split == 'train')[0]; te = np.where(m.split == 'test')[0]
    y_all = np.zeros(len(m), int); y_all[tr] = m.y.values[tr].astype(int); y = y_all[tr]
    oof = np.zeros((len(blend.SEEDS), len(tr), 4)); test = np.zeros((len(te), 4))
    for si, s in enumerate(blend.SEEDS):
        f = folds(y, s)
        for k in range(5):
            va = np.where(f == k)[0]; net = fit(seqs, y_all, tr[f != k], epochs, seed=100 * s + k)
            oof[si, va] = predict(net, seqs, tr[va]); test += predict(net, seqs, te) / 15
        print('seed', s, 'oof macro-F1 (model only, all rows)', round(mf1(y, oof[si].argmax(1)), 4), flush=True)
    print('FINAL', out, blend.evaluate(m, oof).round(4))
    np.savez(f'{BH}/work/probs/{out}.npz', oof=oof, test=test)
    os.makedirs(f'{BH}/work/models', exist_ok=True)
    net = fit(seqs, y_all, tr, epochs, seed=999); torch.save(net.state_dict(), f'{BH}/work/models/{out}_full.pt')
