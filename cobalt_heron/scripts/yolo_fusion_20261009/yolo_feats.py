"""Per-node YOLO evidence: best IoU with a YOLO instance (conf>=0.05), its conf, and max conf among IoU>0.3 matches."""
import json
import numpy as np
from pycocotools import mask as mu


def load(path):
    P = json.load(open(path))
    return {s: [({'size': [2048, 2048], 'counts': x['rle'].encode()}, x['conf']) for x in v if x['conf'] >= 0.05] for s, v in P.items()}


def node_yolo(r, Y):
    n = len(r['rles']); out = np.zeros((n, 3), np.float32)
    items = Y.get(r['stem'], [])
    if n == 0 or not items: return out
    yr = [a for a, _ in items]; yc = np.array([c for _, c in items])
    iou = np.asarray(mu.iou(r['rles'], yr, [0] * len(yr))).reshape(n, len(yr))
    b = iou.argmax(1)
    out[:, 0] = iou.max(1); out[:, 1] = yc[b] * (out[:, 0] > 0.1)
    out[:, 2] = np.where(iou > 0.3, yc[None, :], 0).max(1)
    return out
