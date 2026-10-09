"""Format + overlap check of a submission CSV."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mu

TEST = Path('/tmp/claude-0/-home-user-kraggle-3-in-one/5b333432-508e-5f10-be96-1dd09fd2558d/scratchpad/research/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images')
stems = {p.stem for p in TEST.glob('*.jpeg')}
for path in sys.argv[1:]:
    df = pd.read_csv(path, dtype=str)
    ok = list(df.columns) == ['filament_id', 'segmentation_rle']
    ok &= df.filament_id.is_unique and df.notna().all().all()
    st = df.filament_id.str.rsplit('_', n=1).str[0]
    ok &= set(st) <= stems
    bad_ov = 0; areas = []; small = 0
    for s, g in df.groupby(st):
        occ = np.zeros((2048, 2048), np.uint16)
        for c in g.segmentation_rle:
            m = mu.decode({'size': [2048, 2048], 'counts': c.encode()}); a = int(m.sum()); areas.append(a)
            if a == 0: ok = False
            occ += m
        bad_ov += int((occ > 1).sum())
    ok &= bad_ov == 0
    print(f"{path}: {'OK' if ok else 'FAIL'} rows={len(df)} images_with_preds={st.nunique()}/{len(stems)} overlap_px={bad_ov} "
          f"area_min={min(areas) if areas else 0} area_med={int(np.median(areas)) if areas else 0} n<300={sum(a < 300 for a in areas)}")
