"""Transcript analysis and text member.

1. Structure: group clips by normalised transcript; within groups (excluding the waveform-duplicate clusters), how are
   emotions distributed? (same sentence -> same emotion, or one sentence in several emotions?)
2. Text member: char n-gram TF-IDF + logistic regression, same 3 x 5 folds as blend.py -> work/probs/text_<name>.npz.
"""
import sys, os, re, unicodedata
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend


def norm(t):
    t = unicodedata.normalize('NFC', str(t)).lower()
    t = ''.join(c for c in t if not unicodedata.category(c).startswith('P'))   # drop punctuation, keep vowel signs
    return re.sub(r'\s+', ' ', t).strip()


if __name__ == '__main__':
    name = sys.argv[1]
    m = load_meta(); T = pd.read_csv(f'{BH}/work/transcripts_{name}.csv')
    assert (T.Id.values == m.Id.values).all()
    m['text'] = T.text.fillna('').map(norm)
    tr = (m.split == 'train').values; y = m.y.values[tr].astype(int)

    # 1. structure
    g = m[tr].groupby('text')
    print('train clips', tr.sum(), 'distinct texts', g.ngroups)
    sizes = g.size(); multi = sizes[sizes > 1]
    print('texts shared by >1 train clip:', len(multi), 'covering', multi.sum(), 'clips')
    # same text, different duplicate cluster (independent recordings)
    rows = []
    for t, d in g:
        d = d.drop_duplicates('cl')
        if len(d) > 1:
            rows.append((t, len(d), d.emotion.nunique(), dict(d.emotion.value_counts())))
    S = pd.DataFrame(rows, columns=['text', 'n_recordings', 'n_emotions', 'emotions'])
    print('texts with >1 independent recording:', len(S))
    print('  of which all the same emotion:', (S.n_emotions == 1).sum(), '| 2 emotions:', (S.n_emotions == 2).sum(),
          '| 3:', (S.n_emotions == 3).sum(), '| 4:', (S.n_emotions == 4).sum())
    print(S.sort_values('n_recordings', ascending=False).head(15).to_string())
    te_texts = set(m.text[~tr]); tr_texts = set(m.text[tr])
    print('test clips whose text occurs in train:', sum(t in tr_texts for t in m.text[~tr]), 'of', (~tr).sum())

    # 2. text member
    X = m.text.values; Xtr, Xte = X[tr], X[~tr]
    mk = lambda: make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                               LogisticRegression(C=3, max_iter=2000))
    oof = np.zeros((len(blend.SEEDS), len(y), 4))
    for si, s in enumerate(blend.SEEDS):
        f = folds(y, s)
        for k in range(5):
            va = f == k; oof[si, va] = mk().fit(Xtr[~va], y[~va]).predict_proba(Xtr[va])
    test = mk().fit(Xtr, y).predict_proba(Xte)
    print('TEXT member novel/all', blend.evaluate(m, oof).round(4), '| model-only seed0 F1', round(mf1(y, oof[0].argmax(1)), 4))
    np.savez(f'{BH}/work/probs/text_{name}.npz', oof=oof, test=test)
