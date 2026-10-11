"""BhavVaani: per-layer pooled embeddings from speech foundation models (Kaggle T4).

Writes /kaggle/working/emb_<name>.npz with keys ids, split, mean[N,L,D], std[N,L,D] (float16).
Every model is wrapped in try/except so one failure does not lose the others.
"""
import glob, os, subprocess, sys, time, traceback
import numpy as np, pandas as pd, soundfile as sf, torch

root = sorted(glob.glob('/kaggle/input/**/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
items = [('train', r.Id, f'{root}/train/{r.filename}') for r in tr.itertuples()] + \
        [('test', r.Id, f'{root}/test/{r.filename}') for r in te.itertuples()]
waves = [sf.read(p, dtype='float32')[0] for _, _, p in items]
ids = np.array([i for _, i, _ in items]); split = np.array([s for s, _, _ in items])
dev = 'cuda'
print('root', root, len(items), torch.cuda.get_device_name(0), flush=True)

def save(name, means, stds, extra=None):
    d = dict(ids=ids, split=split, mean=np.stack(means).astype(np.float16), std=np.stack(stds).astype(np.float16))
    if extra: d.update(extra)
    np.savez(f'/kaggle/working/emb_{name}.npz', **d)
    print('saved', name, d['mean'].shape, flush=True)

@torch.no_grad()
def run_ssl(name, repo):
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(dev).eval().half()
    means, stds = [], []
    for w in waves:
        inp = fe(w, sampling_rate=16000, return_tensors='pt')
        inp = {k: (v.to(dev).half() if v.dtype.is_floating_point else v.to(dev)) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True)
        h = torch.stack(out.hidden_states, 0)[:, 0].float()   # L,T,D
        means.append(h.mean(1).cpu().numpy()); stds.append(h.std(1).cpu().numpy())
    save(name, means, stds); del model; torch.cuda.empty_cache()

@torch.no_grad()
def run_whisper(name, repo):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(repo)
    enc = WhisperModel.from_pretrained(repo, torch_dtype=torch.float16).encoder.to(dev).eval()
    means, stds = [], []
    for w in waves:
        f = fe(w, sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
        out = enc(f, output_hidden_states=True)
        T = max(1, int(np.ceil(len(w) / 320)))  # 50 frames / s after the conv stride
        h = torch.stack(out.hidden_states, 0)[:, 0, :T].float()
        means.append(h.mean(1).cpu().numpy()); stds.append(h.std(1).cpu().numpy())
    save(name, means, stds); del enc; torch.cuda.empty_cache()

def run_e2v(name, repo):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'funasr', 'modelscope'], check=True)
    from funasr import AutoModel
    m = AutoModel(model=repo, hub='hf', disable_update=True)
    embs, scores = [], []
    for w in waves:
        r = m.generate(w, granularity='utterance', extract_embedding=True)[0]
        embs.append(np.asarray(r['feats']).reshape(1, -1)); scores.append(np.asarray(r['scores']))
        labels = r['labels']
    E = np.stack(embs)
    save(name, list(E), list(np.zeros_like(E)), dict(scores=np.stack(scores), labels=np.array(labels)))

jobs = [
    ('e2v_plus_large', run_e2v, 'emotion2vec/emotion2vec_plus_large'),
    ('wavlm_large', run_ssl, 'microsoft/wavlm-large'),
    ('whisper_large_v3', run_whisper, 'openai/whisper-large-v3'),
    ('w2vbert2', run_ssl, 'facebook/w2v-bert-2.0'),
    ('hubert_large', run_ssl, 'facebook/hubert-large-ll60k'),
    ('xlsr_300m', run_ssl, 'facebook/wav2vec2-xls-r-300m'),
    ('mms_300m', run_ssl, 'facebook/mms-300m'),
    ('audeering_dim', run_ssl, 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'),
    ('whisper_medium', run_whisper, 'openai/whisper-medium'),
]
for name, fn, repo in jobs:
    t = time.time()
    try:
        fn(name, repo)
    except Exception:
        traceback.print_exc()
    print(f'{name} done in {time.time() - t:.0f}s', flush=True)
