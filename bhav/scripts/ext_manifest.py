"""Manifest of external clips in the four competition classes (path, emotion, source, speaker)."""
import glob, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import BH
X = f'{BH}/work/ext'
MAP = {'anger': 'angry', 'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'}
rows = []
for p in sorted(glob.glob(f'{X}/**/*.wav', recursive=True)):
    r = p.replace(X + '/', ''); src = r.split('/')[0]; parts = r.split('/')
    if src == 'speech-emotion-recognition-hindi':
        emo, spk = parts[-2], 'vish' + parts[2]
    elif src == 'indian-emotional-speech-corpora-iesc':
        emo, spk = parts[-2], 'iesc' + parts[-3]
    elif src == 'multilingual-speech-dataset':
        if parts[2] != 'Hindi':
            continue
        emo = parts[-2]; spk = 'ml' + re.match(r'S(\d+)', parts[-1]).group(1) if re.match(r'S(\d+)', parts[-1]) else 'ml'
    elif src == 'hindi-speech-audio-dataset':
        emo, spk = parts[-1].split('_')[0], 'hsa' + parts[-2]
    else:
        continue
    e = MAP.get(emo.lower())
    if e:
        rows.append(dict(path=p, emotion=e, source=src, speaker=spk))
M = pd.DataFrame(rows); M.to_csv(f'{BH}/work/ext_manifest.csv', index=False)
print(M.groupby(['source', 'emotion']).size().unstack()); print(M.speaker.nunique(), 'speakers', len(M), 'clips')
