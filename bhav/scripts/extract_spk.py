"""ECAPA speaker embeddings (speechbrain/spkrec-ecapa-voxceleb) for competition (+ --ext) clips."""
import sys, os
import numpy as np, pandas as pd, torch, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, EMB, BH
from speechbrain.inference.speaker import EncoderClassifier
enc = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir=f'{BH}/work/hf/ecapa', run_opts={'device': 'cpu'})
m = load_meta(); paths = [f'{DATA}/{s}/{fn}' for s, fn in zip(m.split, m.filename)]
E = []
with torch.no_grad():
    for p in paths:
        w = torch.from_numpy(sf.read(p, dtype='float32')[0])[None]
        E.append(enc.encode_batch(w)[0, 0].numpy())
np.save(f'{EMB}/spk_ecapa.npy', np.stack(E)); print('done', np.stack(E).shape)
