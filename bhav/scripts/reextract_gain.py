"""Re-extract one Whisper encoder's pooled features for the 32-bit PCM clips after scaling them to the median peak of
the 16-bit clips; writes emb_<name>_g.npz = original features with those rows replaced.
python reextract_gain.py <emb name> <hf repo> [target_peak]"""
import sys, os, time
import numpy as np, pandas as pd, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, EMB
from transformers import WhisperFeatureExtractor, WhisperModel
name, repo = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
m = load_meta()
info = [sf.info(f'{DATA}/{s}/{f}').subtype for s, f in zip(m.split, m.filename)]
p32 = np.array([i == 'PCM_32' for i in info])
waves = {i: sf.read(f'{DATA}/{m.split[i]}/{m.filename[i]}', dtype='float32')[0] for i in range(len(m))}
peak16 = np.median([np.abs(waves[i]).max() for i in np.where(~p32)[0]])
target = float(sys.argv[3]) if len(sys.argv) > 3 else peak16
z = dict(np.load(f'{EMB}/emb_{name}.npz', allow_pickle=True)); key = {(s, i): k for k, (s, i) in enumerate(zip(z['split'], z['ids']))}
fe = WhisperFeatureExtractor.from_pretrained(repo); enc = WhisperModel.from_pretrained(repo).encoder.eval()
mean, std = z['mean'].copy(), z['std'].copy(); t = time.time()
with torch.no_grad():
    for n, i in enumerate(np.where(p32)[0]):
        w = waves[i]; w = w * (target / (np.abs(w).max() + 1e-9))
        h = torch.stack(enc(fe(w, sampling_rate=16000, return_tensors='pt').input_features, output_hidden_states=True).hidden_states, 0)[:, 0]
        T = max(1, int(np.ceil(len(w) / 320))); h = h[:, :T]
        k = key[(m.split[i], m.Id[i])]; mean[k] = h.mean(1).numpy(); std[k] = h.std(1).numpy()
        if n % 20 == 0: print(n, f'{time.time()-t:.0f}s', flush=True)
z['mean'], z['std'] = mean.astype(np.float16), std.astype(np.float16)
np.savez(f'{EMB}/emb_{name}_g.npz', **z); print('done', p32.sum(), 'clips, target peak', round(target, 4), round(time.time() - t))
