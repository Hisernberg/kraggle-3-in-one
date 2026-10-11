"""BhavVaani: fine-tune a speech SSL backbone for 4-class Hindi emotion (Kaggle T4).

Head: softmax-weighted sum of all hidden layers -> attentive statistics pooling -> MLP.
5-fold CV on the same StratifiedKFold(seed 0) folds as the local pipeline, then one model on
all training data. Writes oof.npy, test_folds.npy, test_full.npy and the full model to
/kaggle/working (the full model is what gets released on Hugging Face).
"""
import glob, json, math, os, random, time
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoFeatureExtractor, AutoModel

CFG = dict(backbone=os.environ.get('BACKBONE', 'microsoft/wavlm-large'), epochs=12, bs=8, crop_s=6.0,
           lr_bb=2e-5, lr_head=1e-3, wd=0.01, warmup=0.1, seed=0, folds=5, ls=0.05)
CLASSES = ['angry', 'happy', 'neutral', 'sad']
root = sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
y = tr.emotion.map({c: i for i, c in enumerate(CLASSES)}).values
Wtr = [sf.read(f'{root}/train/{f}', dtype='float32')[0] for f in tr.filename]
Wte = [sf.read(f'{root}/test/{f}', dtype='float32')[0] for f in te.filename]
dev = 'cuda'
fe = AutoFeatureExtractor.from_pretrained(CFG['backbone'])
NORM = getattr(fe, 'do_normalize', True)
CROP = int(CFG['crop_s'] * 16000)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prep(w, train):
    if train:
        if len(w) > CROP:
            s = np.random.randint(0, len(w) - CROP + 1); w = w[s:s + CROP]
        w = w * 10 ** (np.random.uniform(-6, 6) / 20)
        if np.random.rand() < 0.3:
            w = w + np.random.randn(len(w)).astype(np.float32) * np.random.uniform(0.001, 0.01) * (np.abs(w).max() + 1e-6)
    if NORM:
        w = (w - w.mean()) / (w.std() + 1e-7)
    return w.astype(np.float32)


def collate(ws):
    L = max(len(w) for w in ws)
    x = np.zeros((len(ws), L), np.float32); mask = np.zeros((len(ws), L), np.int64)
    for i, w in enumerate(ws):
        x[i, :len(w)] = w; mask[i, :len(w)] = 1
    return torch.from_numpy(x), torch.from_numpy(mask)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(CFG['backbone'])
        if hasattr(self.bb, 'freeze_feature_encoder'):
            self.bb.freeze_feature_encoder()
        c = self.bb.config; L = c.num_hidden_layers + 1; D = c.hidden_size
        self.lw = nn.Parameter(torch.zeros(L))
        self.att = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * D), nn.Dropout(0.2), nn.Linear(2 * D, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, 4))

    def forward(self, x, mask):
        out = self.bb(x, attention_mask=mask if self.bb.config.feat_extract_norm == 'layer' else None,
                      output_hidden_states=True)
        H = torch.stack(out.hidden_states, 0)                          # L,B,T,D
        h = (F.softmax(self.lw, 0)[:, None, None, None] * H).sum(0)  # B,T,D
        T = h.shape[1]
        fl = self.bb._get_feat_extract_output_lengths(mask.sum(1)).clamp(max=T)
        fm = (torch.arange(T, device=h.device)[None] < fl[:, None]).float()
        a = self.att(h).squeeze(-1).masked_fill(fm == 0, -1e4).softmax(1)[..., None]
        mu = (a * h).sum(1); sd = ((a * (h - mu[:, None]) ** 2).sum(1) + 1e-6).sqrt()
        return self.head(torch.cat([mu, sd], 1).float())


@torch.no_grad()
def predict(model, W):
    model.eval(); P = []
    for i in range(0, len(W), 8):
        x, mask = collate([prep(w, False) for w in W[i:i + 8]])
        with torch.autocast('cuda', dtype=torch.float16):
            P.append(model(x.to(dev), mask.to(dev)).float().softmax(1).cpu().numpy())
    return np.concatenate(P)


def train(idx, val_idx=None):
    seed_all(CFG['seed']); model = Net().to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith('bb.') and p.requires_grad]
    hd = [p for n, p in model.named_parameters() if not n.startswith('bb.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': CFG['lr_bb']}, {'params': hd, 'lr': CFG['lr_head']}], weight_decay=CFG['wd'])
    steps = CFG['epochs'] * math.ceil(len(idx) / CFG['bs']); warm = int(CFG['warmup'] * steps)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    scaler = torch.cuda.amp.GradScaler(); best = (-1, None)
    for ep in range(CFG['epochs']):
        model.train(); perm = np.random.permutation(idx); t = time.time()
        for b in range(0, len(perm), CFG['bs']):
            bi = perm[b:b + CFG['bs']]
            x, mask = collate([prep(Wtr[i], True) for i in bi])
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x.to(dev), mask.to(dev)), torch.from_numpy(y[bi]).to(dev), label_smoothing=CFG['ls'])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        msg = f'ep {ep} loss {loss.item():.3f} {time.time() - t:.0f}s'
        if val_idx is not None and ep >= CFG['epochs'] // 2:
            pv = predict(model, [Wtr[i] for i in val_idx]); f1 = f1_score(y[val_idx], pv.argmax(1), average='macro')
            msg += f' val {f1:.4f}'
        print(msg, flush=True)
    return model   # last epoch (no checkpoint picking on the validation fold)


skf = StratifiedKFold(CFG['folds'], shuffle=True, random_state=CFG['seed'])
oof = np.zeros((len(tr), 4)); tf = []
for k, (a, b) in enumerate(skf.split(np.zeros(len(y)), y)):
    model = train(a, b); oof[b] = predict(model, [Wtr[i] for i in b]); tf.append(predict(model, Wte))
    print(f'fold {k} f1 {f1_score(y[b], oof[b].argmax(1), average="macro"):.4f}', flush=True)
    del model; torch.cuda.empty_cache()
print('OOF macro-F1', f1_score(y, oof.argmax(1), average='macro'))
np.save('/kaggle/working/oof.npy', oof); np.save('/kaggle/working/test_folds.npy', np.stack(tf))
model = train(np.arange(len(tr))); np.save('/kaggle/working/test_full.npy', predict(model, Wte))
os.makedirs('/kaggle/working/model', exist_ok=True)
torch.save(model.state_dict(), '/kaggle/working/model/model.pt')
json.dump(dict(CFG, classes=CLASSES, sample_rate=16000), open('/kaggle/working/model/config.json', 'w'), indent=1)
pd.DataFrame({'Id': tr.Id}).to_csv('/kaggle/working/train_ids.csv', index=False)
pd.DataFrame({'Id': te.Id}).to_csv('/kaggle/working/test_ids.csv', index=False)
