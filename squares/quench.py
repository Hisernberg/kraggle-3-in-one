"""Penalty-based compression of one packing (CPU, numpy + scipy L-BFGS).

For a target side s < L(n): squares (x, y, t) must lie in [0, s]^2 and not overlap.
Penalty = sum over near pairs of (SAT penetration depth)^2 + containment violation^2.
After the penalty reaches ~0 the packing is made strictly feasible by scaling the
centres about the box centre by (1 + eta) (separates residual overlaps), then L is re-measured.
Usage: python quench.py merged.csv n [steps] [seconds]
"""
import sys, time, math
import numpy as np, pandas as pd
from scipy.optimize import minimize

H = 0.5


def corners(x, y, t):
    c, s = np.cos(t), np.sin(t)
    ox = np.array([H, -H, -H, H]); oy = np.array([H, H, -H, -H])
    cx = x[:, None] + c[:, None] * ox - s[:, None] * oy
    cy = y[:, None] + s[:, None] * ox + c[:, None] * oy
    return cx, cy


def side(x, y, t):
    cx, cy = corners(x, y, t)
    return max(cx.max() - cx.min(), cy.max() - cy.min())


def pairs_near(x, y, cut=1.5):
    from scipy.spatial import cKDTree
    return np.array(sorted(cKDTree(np.c_[x, y]).query_pairs(cut)), dtype=int).reshape(-1, 2)


def penalty(v, n, s, P, need_grad=True):
    x, y, t = v[:n], v[n:2 * n], v[2 * n:]
    c, sn = np.cos(t), np.sin(t)
    hx = H * (np.abs(c) + np.abs(sn))  # half-extent
    # containment
    lo = hx - x; hi = x + hx - s; loy = hx - y; hiy = y + hx - s
    f = 0.0
    g = np.zeros_like(v)
    for arr, sign, off in ((lo, -1, 0), (hi, 1, 0), (loy, -1, n), (hiy, 1, n)):
        m = arr > 0
        f += np.sum(arr[m] ** 2)
        g[off:off + n][m] += 2 * arr[m] * sign
        dh = H * (np.sign(c) * -sn + np.sign(sn) * c)  # d hx / dt
        g[2 * n:][m] += 2 * arr[m] * dh[m]
    if len(P):
        i, j = P[:, 0], P[:, 1]
        dx = x[j] - x[i]; dy = y[j] - y[i]
        best = np.full(len(P), np.inf); bax = np.zeros(len(P)); bay = np.zeros(len(P)); bk = np.zeros(len(P), int)
        axes = [(c[i], sn[i]), (-sn[i], c[i]), (c[j], sn[j]), (-sn[j], c[j])]
        for k, (ax, ay) in enumerate(axes):
            ri = H * (np.abs(ax * c[i] + ay * sn[i]) + np.abs(-ax * sn[i] + ay * c[i]))
            rj = H * (np.abs(ax * c[j] + ay * sn[j]) + np.abs(-ax * sn[j] + ay * c[j]))
            pen = ri + rj - np.abs(ax * dx + ay * dy)
            upd = pen < best
            best = np.where(upd, pen, best); bax = np.where(upd, ax, bax); bay = np.where(upd, ay, bay); bk = np.where(upd, k, bk)
        m = best > 0
        f += np.sum(best[m] ** 2)
        if need_grad and m.any():
            # gradient w.r.t. positions only (angles: numerical-free approximation, ignore axis rotation)
            pd_ = 2 * best[m]
            sgn = np.sign(bax[m] * dx[m] + bay[m] * dy[m])
            gx = pd_ * sgn * bax[m]; gy = pd_ * sgn * bay[m]
            np.add.at(g[:n], i[m], gx); np.add.at(g[:n], j[m], -gx)
            np.add.at(g[n:2 * n], i[m], gy); np.add.at(g[n:2 * n], j[m], -gy)
    return f, g


def quench(x, y, t, s, iters=3000):
    n = len(x)
    v = np.r_[x, y, t]
    for rnd in range(6):
        P = pairs_near(v[:n], v[n:2 * n])
        r = minimize(penalty, v, args=(n, s, P), jac=True, method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-16})
        v = r.x
        if r.fun < 1e-24: break
    return v, r.fun


def make_feasible(v, n):
    """Scale centres about the centroid until float SAT says every pair is separated (with margin)."""
    x, y, t = v[:n].copy(), v[n:2 * n].copy(), v[2 * n:].copy()
    P = pairs_near(x, y)
    mx, my = x.mean(), y.mean()
    for eta in [0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
        xs = mx + (x - mx) * (1 + eta); ys = my + (y - my) * (1 + eta)
        f, _ = penalty(np.r_[xs, ys, t], n, 1e9, P, need_grad=False)
        # check min separation over pairs: need max penetration < -1e-11 or exact; use penetration
        if pen_max(xs, ys, t, P) < -2e-12:
            return xs, ys, t
    return None


def pen_max(x, y, t, P):
    if not len(P): return -1
    i, j = P[:, 0], P[:, 1]
    c, sn = np.cos(t), np.sin(t)
    dx = x[j] - x[i]; dy = y[j] - y[i]
    best = np.full(len(P), np.inf)
    for ax, ay in [(c[i], sn[i]), (-sn[i], c[i]), (c[j], sn[j]), (-sn[j], c[j])]:
        ri = H * (np.abs(ax * c[i] + ay * sn[i]) + np.abs(-ax * sn[i] + ay * c[i]))
        rj = H * (np.abs(ax * c[j] + ay * sn[j]) + np.abs(-ax * sn[j] + ay * c[j]))
        best = np.minimum(best, ri + rj - np.abs(ax * dx + ay * dy))
    return best.max()


def run(path, n, steps=8, secs=120, seed=0):
    d = pd.read_csv(path); q = d[d.n == n].sort_values('square_id')
    x, y, t = q.x.values.astype(float), q.y.values.astype(float), q.angle.values.astype(float)
    cx, cy = corners(x, y, t); x = x - cx.min(); y = y - cy.min()
    L0 = side(x, y, t); bestL, best = L0, None
    rng = np.random.default_rng(seed); t0 = time.time()
    step = 1e-4
    cur = (x, y, t)
    while time.time() - t0 < secs:
        s = bestL - step
        xx, yy, tt = cur
        jit = 1e-3 * step / 1e-4
        v0 = np.r_[xx * s / bestL + rng.normal(0, jit, n), yy * s / bestL + rng.normal(0, jit, n), tt + rng.normal(0, jit * 0.1, n)]
        v, f = quench(v0[:n], v0[n:2 * n], v0[2 * n:], s)
        if f < 1e-22:
            res = make_feasible(v, n)
            if res is not None:
                Ln = side(*res)
                if Ln < bestL - 1e-12:
                    bestL, best, cur = Ln, res, res
                    print(f'n={n} improved L {Ln:.12f} (gain {L0 - Ln:.3e})', flush=True)
                    step *= 1.5; continue
        step /= 2
        if step < 1e-9: step = 1e-5
    return L0, bestL, best


if __name__ == '__main__':
    path, n = sys.argv[1], int(sys.argv[2]); secs = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    L0, L1, best = run(path, n, secs=secs)
    print('RESULT', n, L0, L1, L0 - L1)
    if best is not None:
        x, y, t = best
        pd.DataFrame({'n': n, 'square_id': range(n), 'x': [repr(float(a)) for a in x], 'y': [repr(float(a)) for a in y], 'angle': [repr(float(a)) for a in t]}).to_csv(f'cand_{n}.csv', index=False)
