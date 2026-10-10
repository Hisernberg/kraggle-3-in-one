"""BhavVaani extraction round 2 (Kaggle T4): more Whisper encoders, external clips, frame-level features.

* emb_<name>.npz         competition clips, per-layer mean/std pooled (same format as round 1)
* emb_ext_<name>.npz     external Hindi emotion clips (dataset sources attached), manifest order of ext.csv
* frames_whisper_large_v3.npz  last-layer frames (float16, concatenated) + offsets, competition clips
"""
import glob, os, time, traceback
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
from transformers import WhisperFeatureExtractor, WhisperModel

root = sorted(glob.glob('/kaggle/input/**/SRCASW-BhavVaani/train.csv', recursive=True))[0].rsplit('/', 1)[0]
tr = pd.read_csv(f'{root}/train.csv'); te = pd.read_csv(f'{root}/test.csv')
items = [('train', r.Id, f'{root}/train/{r.filename}') for r in tr.itertuples()] + \
        [('test', r.Id, f'{root}/test/{r.filename}') for r in te.itertuples()]
waves = [sf.read(p, dtype='float32')[0] for _, _, p in items]
ids = np.array([i for _, i, _ in items]); split = np.array([s for s, _, _ in items])

MAP = {'anger': 'angry', 'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'}
ext = []
for p in sorted(glob.glob('/kaggle/input/**/*.wav', recursive=True)):
    if 'SRCASW-BhavVaani' in p:
        continue
    parts = p.split('/')
    if 'multilingual' in p.lower() and '/Hindi/' not in p:
        continue
    emo = parts[-1].split('_')[0] if 'Hindi Speech Audio' in p else parts[-2]
    if emo.lower() in MAP:
        ext.append((p, MAP[emo.lower()]))
pd.DataFrame(ext, columns=['path', 'emotion']).to_csv('/kaggle/working/ext.csv', index=False)
print('ext clips', len(ext), flush=True)
dev = 'cuda'


@torch.no_grad()
def whisper(repo, W, frames=False):
    fe = WhisperFeatureExtractor.from_pretrained(repo)
    enc = WhisperModel.from_pretrained(repo, torch_dtype=torch.float16).encoder.to(dev).eval()
    means, stds, fr = [], [], []
    for w in W:
        f = fe(w[:30 * 16000], sampling_rate=16000, return_tensors='pt').input_features.to(dev).half()
        out = enc(f, output_hidden_states=True)
        T = max(1, int(np.ceil(min(len(w), 30 * 16000) / 320)))
        h = torch.stack(out.hidden_states, 0)[:, 0, :T].float()
        means.append(h.mean(1).cpu().numpy()); stds.append(h.std(1).cpu().numpy())
        if frames:
            fr.append(h[-1].half().cpu().numpy())
    del enc; torch.cuda.empty_cache()
    return np.stack(means).astype(np.float16), np.stack(stds).astype(np.float16), fr


jobs = [  # (name, repo, ext, frames)
    ('whisper_large_v3', 'openai/whisper-large-v3', True, True),
    ('whisper_medium', 'openai/whisper-medium', True, False),
    ('whisper_hi_large_v2', 'vasista22/whisper-hindi-large-v2', False, False),
    ('whisper_vaani_large_v3', 'ARTPARK-IISc/whisper-large-v3-vaani-hindi', False, False),
    ('whisper_large_v2', 'openai/whisper-large-v2', False, False),
    ('whisper_large_v3_turbo', 'openai/whisper-large-v3-turbo', False, False),
    ('whisper_hinglish_prime', 'Oriserve/Whisper-Hindi2Hinglish-Prime', False, False),
    ('whisper_large_v1', 'openai/whisper-large', False, False),
    ('whisper_hi_medium', 'vasista22/whisper-hindi-medium', False, False),
    ('whisper_collabora_hi_large_v2', 'collabora/whisper-large-v2-hindi', False, False),
]
Wext = None
for name, repo, do_ext, do_frames in jobs:
    t = time.time()
    try:
        if name not in ('whisper_large_v3', 'whisper_medium'):
            M, S, _ = whisper(repo, waves)
            np.savez(f'/kaggle/working/emb_{name}.npz', ids=ids, split=split, mean=M, std=S)
        elif do_frames:
            M, S, fr = whisper(repo, waves, frames=True)
            off = np.cumsum([0] + [len(x) for x in fr])
            np.savez(f'/kaggle/working/frames_{name}.npz', ids=ids, split=split, frames=np.concatenate(fr), offsets=off)
        if do_ext:
            if Wext is None:
                Wext = [librosa.load(p, sr=16000)[0] for p, _ in ext]
            M, S, _ = whisper(repo, Wext)
            np.savez(f'/kaggle/working/emb_ext_{name}.npz', ids=np.arange(len(ext)), split=np.array(['ext'] * len(ext)), mean=M, std=S)
    except Exception:
        traceback.print_exc()
    print(f'{name} done in {time.time() - t:.0f}s', flush=True)
