"""Fast float per-n side length L and score for a submission CSV (screening only; certify with validate.py)."""
import sys, numpy as np, pandas as pd
def per_n(path):
    d = pd.read_csv(path, dtype={'x': str, 'y': str, 'angle': str})
    x = d.x.astype(float).values; y = d.y.astype(float).values; a = d.angle.astype(float).values
    c, s = np.cos(a), np.sin(a)
    hx = 0.5 * (np.abs(c) + np.abs(s))  # half extent of rotated unit square on each axis
    d['x0'] = x - hx; d['x1'] = x + hx; d['y0'] = y - hx; d['y1'] = y + hx
    g = d.groupby('n').agg(x0=('x0', 'min'), x1=('x1', 'max'), y0=('y0', 'min'), y1=('y1', 'max'), cnt=('n', 'size'))
    g['L'] = np.maximum(g.x1 - g.x0, g.y1 - g.y0)
    g['score'] = g.L / np.sqrt(g.index.values) - 1
    return g[['L', 'score', 'cnt']]
if __name__ == '__main__':
    for p in sys.argv[1:]:
        g = per_n(p); print(p, len(g), round(g.score.sum(), 7))
