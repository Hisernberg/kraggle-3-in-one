"""eGeMAPSv02 functionals (88) + duration / level statistics for all competition clips -> work/emb/emb_egemaps.npz."""
import sys, os
import numpy as np, soundfile as sf, opensmile
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, DATA, EMB

def feats(p):
    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    w, sr = sf.read(p, dtype='float32')
    f = sm.process_signal(w, sr).values[0]
    a = np.abs(w); extra = [len(w) / sr, np.log(np.sqrt((w ** 2).mean()) + 1e-6), np.log(a.max() + 1e-6), (a > 0.02).mean()]
    return np.r_[f, extra].astype(np.float32)

m = load_meta()
X = np.array(Parallel(n_jobs=2)(delayed(feats)(f'{DATA}/{s}/{fn}') for s, fn in zip(m.split, m.filename)))
X = np.nan_to_num(X)[:, None, :]
np.savez(f'{EMB}/emb_egemaps.npz', ids=m.Id.values, split=m.split.values.astype(str), mean=X, std=np.zeros_like(X))
print('done', X.shape)
