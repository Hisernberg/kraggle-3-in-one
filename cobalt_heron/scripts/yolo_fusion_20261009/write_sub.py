"""Write a submission from selected nodes: greedy disjoint by score, area floor, CSV."""
import numpy as np, pandas as pd
from pycocotools import mask as mu


def resolve(rles, scores, floor=0, one_piece=False):
    """greedy by score desc: remove already-occupied pixels, drop if < floor; returns list of rles."""
    from scipy import ndimage as ndi
    order = np.argsort(-np.asarray(scores)); occ = None; out = []
    for i in order:
        m = mu.decode(rles[i]).astype(bool)
        if occ is None: occ = np.zeros_like(m)
        m &= ~occ
        if one_piece and m.any():
            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            if n > 1:
                sz = np.bincount(lab.ravel()); sz[0] = 0; m = lab == sz.argmax()
        if m.sum() < max(floor, 1): continue
        occ |= m; out.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    return out


def to_rows(stem, rles):
    return [{'filament_id': f'{stem}_{k}', 'segmentation_rle': r['counts'].decode('ascii')} for k, r in enumerate(rles, 1)]


def write(rows, path):
    pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']).to_csv(path, index=False)
