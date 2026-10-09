"""tar of float16 npz prob maps -> dir of uint8 npz (p: 2x1024x1024)."""
import sys, tarfile, io, numpy as np
from pathlib import Path
tp, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n = 0
with tarfile.open(tp) as t:
    for m in t:
        if not m.name.endswith('.npz'): continue
        p = np.load(io.BytesIO(t.extractfile(m).read()))['p'].astype(np.float32)
        np.savez_compressed(out / Path(m.name).name, p=np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)); n += 1
print(tp, '->', out, n)
