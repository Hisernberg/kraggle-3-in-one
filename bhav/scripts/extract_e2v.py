"""emotion2vec embeddings + 9-class scores on CPU: python extract_e2v.py <name> <hf repo>."""
import sys, os, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, EMB
from funasr import AutoModel
name, repo = sys.argv[1], sys.argv[2]
m = load_meta(); model = AutoModel(model=repo, hub='hf', disable_update=True, device='cpu')
E, Sc = [], []; t = time.time()
for k, (s, fn) in enumerate(zip(m.split, m.filename)):
    r = model.generate(f'{DATA}/{s}/{fn}', granularity='utterance', extract_embedding=True, disable_pbar=True)[0]
    E.append(np.asarray(r['feats']).reshape(1, -1)); Sc.append(np.asarray(r['scores'])); labels = r['labels']
    if k % 200 == 0: print(k, f'{time.time()-t:.0f}s', flush=True)
E = np.stack(E)
np.savez(f'{EMB}/emb_{name}.npz', ids=m.Id.values, split=m.split.values.astype(str), mean=E.astype(np.float16),
         std=np.zeros_like(E, dtype=np.float16), scores=np.stack(Sc), labels=np.array(labels))
print('done', time.time() - t, labels)
