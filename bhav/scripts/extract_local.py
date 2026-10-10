"""CPU fallback of the Kaggle extraction kernel for wav2vec-style models.

python extract_local.py <name> <repo> [--ext]   (--ext: clips of work/ext_manifest.csv -> emb_ext_<name>.npz)
"""
import sys, os, time
import numpy as np, pandas as pd, soundfile as sf, torch, librosa
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, EMB, BH
from transformers import AutoFeatureExtractor, AutoModel
name, repo = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
EXT = '--ext' in sys.argv
if EXT:
    m = pd.read_csv(f'{BH}/work/ext_manifest.csv'); paths = list(m.path)
else:
    m = load_meta(); paths = [f'{DATA}/{s}/{fn}' for s, fn in zip(m.split, m.filename)]
fe = AutoFeatureExtractor.from_pretrained(repo); model = AutoModel.from_pretrained(repo).eval()
means, stds = [], []; t = time.time()
with torch.no_grad():
    for k, p in enumerate(paths):
        w = librosa.load(p, sr=16000)[0] if EXT else sf.read(p, dtype='float32')[0]
        out = model(**fe(w, sampling_rate=16000, return_tensors='pt'), output_hidden_states=True)
        h = torch.stack(out.hidden_states, 0)[:, 0]
        means.append(h.mean(1).numpy()); stds.append(h.std(1).numpy())
        if k % 100 == 0: print(k, f'{time.time()-t:.0f}s', flush=True)
np.savez(f'{EMB}/emb_ext_{name}.npz' if EXT else f'{EMB}/emb_{name}.npz',
         ids=np.arange(len(m)) if EXT else m.Id.values, split=np.array(['ext'] * len(m)) if EXT else m.split.values.astype(str),
         mean=np.stack(means).astype(np.float16), std=np.stack(stds).astype(np.float16))
print('done', time.time() - t)
