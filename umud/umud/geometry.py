"""Muscle-architecture geometry from aponeurosis / fascicle probability maps.

All inputs are on the B-mode crop (see scale.detect_scale). Every function returns
several estimator variants (e.g. MT between inner edges vs. band centrelines) so
that the final choice can be validated rather than hard-coded.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------- aponeuroses
def _band_profiles(mask: np.ndarray, prob: np.ndarray, lab: np.ndarray, k: int):
    """Per-column upper edge, lower edge and prob-weighted centre of component k."""
    ys, xs = np.where(lab == k)
    W = mask.shape[1]
    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    cen = np.full(W, np.nan)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    bounds = np.flatnonzero(np.diff(xs)) + 1
    for seg_x, seg_y in zip(np.split(xs, bounds), np.split(ys, bounds)):
        x = seg_x[0]
        top[x] = seg_y.min()
        bot[x] = seg_y.max()
        w = prob[seg_y, x]
        cen[x] = float((seg_y * w).sum() / max(w.sum(), 1e-6))
    return top, bot, cen


def _fit(x: np.ndarray, y: np.ndarray, deg: int = 1):
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return None
    xs, ys = x[ok], y[ok]
    # robust: two passes dropping the worst 10% residuals
    c = np.polyfit(xs, ys, deg)
    for _ in range(2):
        r = np.abs(np.polyval(c, xs) - ys)
        keep = r <= np.percentile(r, 90) + 1e-6
        if keep.sum() >= 10:
            c = np.polyfit(xs[keep], ys[keep], deg)
    return c


DEEP_RULE = "hybrid"  # "widest" (original), "nearest" (first substantial band below the superficial one),
# or "hybrid": nearest unless that implies MT < 12 mm (then widest). Nearest matches the superficial-muscle
# protocol; widest picked a deeper boundary on 3-band images (e.g. MT 40 mm vs ~25 mm).
HYBRID_MIN_MT_MM = 12.0


def find_aponeuroses(apo_prob: np.ndarray, px_per_mm: float, thr: float = 0.5):
    """Return dict with superficial/deep line fits (edges + centre) or None."""
    H, W = apo_prob.shape
    m = (apo_prob >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    cands = []
    for k in range(1, n):
        span = st[k, cv2.CC_STAT_WIDTH]
        if span < 0.2 * W or st[k, cv2.CC_STAT_AREA] < 0.002 * H * W:
            continue
        cands.append(k)
    if len(cands) < 2:
        return None
    x = np.arange(W, dtype=float)
    bands = []
    for k in cands:
        top, bot, cenl = _band_profiles(m, apo_prob, lab, k)
        c_top, c_bot, c_cen = _fit(x, top), _fit(x, bot), _fit(x, cenl)
        if c_cen is None:
            continue
        bands.append(dict(k=k, top=c_top, bot=c_bot, cen=c_cen, span=st[k, cv2.CC_STAT_WIDTH], top_raw=top, bot_raw=bot,
                          ymid=float(np.polyval(c_cen, W / 2)),
                          mass=float(apo_prob[lab == k].sum())))
    bands.sort(key=lambda b: b["ymid"])
    if len(bands) < 2:
        return None
    # superficial: upper-most substantial band; deep: strongest band >= 5 mm below it
    sup = bands[0]
    below = [b for b in bands[1:] if b["ymid"] - sup["ymid"] >= 5 * px_per_mm]
    if not below:
        return None
    widest = max(below, key=lambda b: b["span"])
    wide = [b for b in below if b["span"] >= 0.4 * W] or below
    nearest = min(wide, key=lambda b: b["ymid"])
    if DEEP_RULE == "nearest":
        deep = nearest
    elif DEEP_RULE == "hybrid":
        top = nearest["top"] if nearest["top"] is not None else nearest["cen"]
        bot = sup["bot"] if sup["bot"] is not None else sup["cen"]
        inner_mm = (np.polyval(top, W / 2) - np.polyval(bot, W / 2)) / px_per_mm
        deep = nearest if inner_mm >= HYBRID_MIN_MT_MM else widest
    else:
        deep = widest
    return dict(sup=sup, deep=deep, n_bands=len(bands), W=W, H=H)


def _perp_dist(c_deep, c_sup, xs):
    """Distance from points on the deep line to the superficial line, measured perpendicular."""
    out = []
    a, b = c_sup[0], c_sup[1]  # y = a x + b
    for x0 in xs:
        y0 = np.polyval(c_deep, x0)
        out.append(abs(a * x0 - y0 + b) / math.sqrt(a * a + 1))
    return np.asarray(out)


def muscle_thickness(apo: dict, px_per_mm: float) -> dict:
    W = apo["W"]
    xs = np.array([0.25, 0.5, 0.75]) * W
    s, d = apo["sup"], apo["deep"]
    res = {}
    for name, cs, cd in (("inner", s["bot"], d["top"]), ("center", s["cen"], d["cen"]), ("outer", s["top"], d["bot"])):
        if cs is None or cd is None:
            res[f"mt_{name}"] = np.nan
            continue
        res[f"mt_{name}"] = float(_perp_dist(cd, cs, xs).mean() / px_per_mm)
        res[f"mt_{name}_vert"] = float(np.mean(np.polyval(cd, xs) - np.polyval(cs, xs)) / px_per_mm)
    res["sup_thick"] = float(np.mean(np.polyval(s["bot"], xs) - np.polyval(s["top"], xs)) / px_per_mm) if s["bot"] is not None and s["top"] is not None else np.nan
    res["deep_thick"] = float(np.mean(np.polyval(d["bot"], xs) - np.polyval(d["top"], xs)) / px_per_mm) if d["bot"] is not None and d["top"] is not None else np.nan
    res["deep_angle"] = float(math.degrees(math.atan(d["cen"][0])))
    res["sup_angle"] = float(math.degrees(math.atan(s["cen"][0])))
    return res


# --------------------------------------------------------------------------- fascicles
def _angle_between(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180
    return min(d, 180 - d)


def _intersect(m, c, line):
    """Intersection x of y = m x + c with y = line[0] x + line[1]."""
    den = m - line[0]
    if abs(den) < 1e-9:
        return None
    return (line[1] - c) / den


def fascicles(fasc_prob: np.ndarray, apo: dict, px_per_mm: float, thr: float = 0.35,
              min_len_mm: float = 3.0) -> dict:
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    xs = np.arange(W)
    y_sup = np.polyval(s["bot"] if s["bot"] is not None else s["cen"], xs)
    y_deep = np.polyval(d["top"] if d["top"] is not None else d["cen"], xs)
    yy = np.arange(H)[:, None]
    margin = 0.5 * px_per_mm
    region = (yy > y_sup[None] + margin) & (yy < y_deep[None] - margin)
    p = np.where(region, fasc_prob, 0).astype(np.float32)
    m = (p >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    deep_ang = math.degrees(math.atan(d["cen"][0]))
    sup_line = s["bot"] if s["bot"] is not None else s["cen"]
    deep_line = d["top"] if d["top"] is not None else d["cen"]
    recs = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 10:
            continue
        ys, xk = np.where(lab == k)
        pts = np.stack([xk, ys], 1).astype(float)
        w = p[ys, xk]
        mu = (pts * w[:, None]).sum(0) / w.sum()
        q = pts - mu
        cov = (q * w[:, None]).T @ q / w.sum()
        ev, vec = np.linalg.eigh(cov)
        v = vec[:, -1]
        length = 4 * math.sqrt(max(ev[-1], 0))  # ~ extent of a uniform segment
        if length < min_len_mm * px_per_mm:
            continue
        elong = math.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9))
        if elong < 3:
            continue
        ang = math.degrees(math.atan2(v[1], v[0]))
        pa = _angle_between(ang, deep_ang)
        if not 2 <= pa <= 60:
            continue
        slope = v[1] / v[0] if abs(v[0]) > 1e-9 else 1e9
        c0 = mu[1] - slope * mu[0]
        xu, xl = _intersect(slope, c0, sup_line), _intersect(slope, c0, deep_line)
        fl = np.nan
        if xu is not None and xl is not None:
            yu, yl = slope * xu + c0, slope * xl + c0
            fl = math.hypot(xu - xl, yu - yl) / px_per_mm
        recs.append((pa, fl, length, float(w.mean()), ang))
    out = dict(n_fasc=len(recs))
    if not recs:
        return out
    R = np.array(recs)
    pa, fl, ln, conf = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    wt = ln * conf
    # direction sanity: fascicles should share one orientation sign relative to the deep apo
    rel = ((R[:, 4] - deep_ang + 90) % 180) - 90
    sign = np.sign(np.average(np.sign(rel), weights=wt))
    keep = np.sign(rel) == sign if sign != 0 else np.ones(len(R), bool)
    if keep.sum() >= 1:
        pa, fl, ln, wt = pa[keep], fl[keep], ln[keep], wt[keep]
    out["pa_med"] = float(np.median(pa))
    out["pa_wmean"] = float(np.average(pa, weights=wt))
    out["pa_wmed"] = _wmedian(pa, wt)
    top = np.argsort(-wt)[:5]
    out["pa_top5"] = float(np.median(pa[top]))
    ok = np.isfinite(fl) & (fl > 10) & (fl < 300)
    if ok.any():
        out["fl_med"] = float(np.median(fl[ok]))
        out["fl_wmed"] = _wmedian(fl[ok], wt[ok])
        t5 = [i for i in top if ok[i]]
        out["fl_top5"] = float(np.median(fl[t5])) if t5 else np.nan
    out["n_fasc_kept"] = int(len(pa))
    out["fasc_sign"] = float(sign)
    return out


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def orientation_pa(fasc_prob: np.ndarray, apo: dict, px_per_mm: float) -> float:
    """PA from the dominant structure-tensor orientation of the fascicle map inside the muscle."""
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    xs = np.arange(W)
    y_sup = np.polyval(s["cen"], xs)
    y_deep = np.polyval(d["cen"], xs)
    yy = np.arange(H)[:, None]
    region = (yy > y_sup[None] + px_per_mm) & (yy < y_deep[None] - px_per_mm)
    f = fasc_prob.astype(np.float32) / 255.0 if fasc_prob.dtype == np.uint8 else fasc_prob.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, 1.5 * px_per_mm)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # line direction
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    w = (coh * f)[region]
    if w.sum() <= 0:
        return np.nan
    ang = np.degrees(theta[region])
    deep_ang = math.degrees(math.atan(d["cen"][0]))
    rel = ((ang - deep_ang + 90) % 180) - 90
    hist, edges = np.histogram(rel, bins=180, range=(-90, 90), weights=w)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    i = int(np.argmax(hist))
    return float(abs(0.5 * (edges[i] + edges[i + 1])))


def _wfit(x, y, w, iters=3):
    """Weighted robust line y = a + b x (drops residuals > 2.5 weighted MADs); returns (a, b) or None."""
    keep = np.ones(len(x), bool)
    ab = None
    for _ in range(iters):
        if keep.sum() < 3 or np.ptp(x[keep]) < 0.15:
            return ab
        W = w[keep]
        X = np.stack([np.ones(keep.sum()), x[keep]], 1)
        A = X.T @ (X * W[:, None])
        ab = np.linalg.solve(A + 1e-9 * np.eye(2), X.T @ (W * y[keep]))
        r = np.abs(y - (ab[0] + ab[1] * x))
        mad = _wmedian(r[keep], W) + 1e-6
        keep = r <= 2.5 * 1.4826 * mad
    return ab


def fascicles_v2(fasc_prob: np.ndarray, apo: dict, px_per_mm: float, thr: float = 0.35,
                 min_len_mm: float = 3.0) -> dict:
    """Depth-resolved fascicle estimators (v2), mimicking the manual protocol.

    Raters measure PA with the angle tool at fascicle *insertions* into the deep aponeurosis, and FL along the
    fascicle path (segmented line), extrapolated straight to the aponeuroses. Fascicles curve, so the angle near
    the deep aponeurosis differs from the mean fragment angle. Each fragment gets a normalised depth
    d = 0 (superficial inner edge) .. 1 (deep inner edge) and its angle to the deep aponeurosis; a weighted
    robust line angle(d) gives the insertion angle at d = 1 and a curved-path FL.
    """
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    sup_line = s["bot"] if s["bot"] is not None else s["cen"]
    deep_line = d["top"] if d["top"] is not None else d["cen"]
    xs = np.arange(W)
    y_sup, y_deep = np.polyval(sup_line, xs), np.polyval(deep_line, xs)
    yy = np.arange(H)[:, None]
    margin = 0.5 * px_per_mm
    region = (yy > y_sup[None] + margin) & (yy < y_deep[None] - margin)
    p = np.where(region, fasc_prob, 0).astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p >= thr).astype(np.uint8), 8)
    deep_ang = math.degrees(math.atan(deep_line[0]))
    sup_ang = math.degrees(math.atan(sup_line[0]))
    R = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 10:
            continue
        ys, xk = np.where(lab == k)
        pts = np.stack([xk, ys], 1).astype(float)
        w = p[ys, xk]
        mu = (pts * w[:, None]).sum(0) / w.sum()
        q = pts - mu
        ev, vec = np.linalg.eigh((q * w[:, None]).T @ q / w.sum())
        v = vec[:, -1]
        length = 4 * math.sqrt(max(ev[-1], 0))
        if length < min_len_mm * px_per_mm or math.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9)) < 3:
            continue
        ang = math.degrees(math.atan2(v[1], v[0]))
        rel = ((ang - deep_ang + 90) % 180) - 90  # signed angle to the deep aponeurosis
        if not 2 <= abs(rel) <= 60:
            continue
        xc = int(np.clip(round(mu[0]), 0, W - 1))
        th = y_deep[xc] - y_sup[xc]
        if th <= 1:
            continue
        dc = (mu[1] - y_sup[xc]) / th
        ends = [mu - 0.5 * length * v, mu + 0.5 * length * v]
        lo = max(ends, key=lambda e: e[1])
        xl = int(np.clip(round(lo[0]), 0, W - 1))
        dlo = (lo[1] - y_sup[xl]) / max(y_deep[xl] - y_sup[xl], 1)
        rel_s = ((ang - sup_ang + 90) % 180) - 90
        axis = (1 - min(max(dc, 0), 1)) * sup_ang + min(max(dc, 0), 1) * deep_ang  # local muscle axis
        rel_ax = ((ang - axis + 90) % 180) - 90
        R.append((rel, rel_s, dc, dlo, length, float(w.mean()), mu[0], rel_ax))
    out = {"v2_n": len(R)}
    if len(R) < 2:
        return out
    R = np.array(R)
    wt = R[:, 4] * R[:, 5]
    sign = np.sign(np.average(np.sign(R[:, 0]), weights=wt)) or 1.0
    keep = np.sign(R[:, 0]) == sign
    R, wt = R[keep], wt[keep]
    if len(R) < 2:
        return out
    pa, pa_s, dc, dlo = np.abs(R[:, 0]), np.abs(R[:, 1]), np.clip(R[:, 2], 0, 1), np.clip(R[:, 3], 0, 1)
    for name, sel in (("lowhalf", dc >= 0.5), ("low3", dc >= 2 / 3), ("uphalf", dc < 0.5), ("insert", dlo >= 0.85)):
        if sel.sum() >= 1:
            out[f"v2_pa_{name}"] = _wmedian(pa[sel], wt[sel])
    ab = _wfit(dc, pa, wt)
    if ab is not None:
        a, b = float(ab[0]), float(np.clip(ab[1], -20, 20))
        out.update(v2_pa_fit1=a + b, v2_pa_fit0=a, v2_pa_fitmid=a + 0.5 * b, v2_pa_slope=b,
                   v2_pa_fit09=a + 0.9 * b)
        # curved-path FL through the middle of the image: thickness along the local normal / sin(angle(d))
        xm = W / 2
        th = (np.polyval(deep_line, xm) - np.polyval(sup_line, xm)) * math.cos(math.radians(0.5 * (deep_ang + sup_ang)))
        dd = np.linspace(0, 1, 41)
        phi = np.clip(a + b * dd, 3, 80)
        out["v2_fl_curve"] = float(np.trapezoid(1 / np.sin(np.radians(phi)), dd) * th / px_per_mm)
        out["v2_fl_chord_mid"] = float(th / px_per_mm / math.sin(math.radians(max(a + 0.5 * b, 3))))
        out["v2_fl_chord_deep"] = float(th / px_per_mm / math.sin(math.radians(max(a + b, 3))))
    out["v2_fasc_span"] = float(np.ptp(dc))
    if ab is not None:
        out.update(streamline_fl(fasc_prob, apo, px_per_mm, a, b, sign))
    # v2b: angles relative to the local muscle axis (interpolated between both aponeuroses by depth), so that
    # diverging aponeuroses do not bend the extrapolated fascicle towards the superficial one
    pax = np.abs(R[:, 7])
    abx = _wfit(dc, pax, wt)
    if abx is not None:
        ax_, bx_ = float(abx[0]), float(np.clip(abx[1], -20, 20))
        out.update(v2b_pa_fit1=ax_ + bx_, v2b_pa_fitmid=ax_ + 0.5 * bx_)
        out.update({k.replace("v2_", "v2b_"): v for k, v in
                    streamline_fl(fasc_prob, apo, px_per_mm, ax_, bx_, sign, axis_rel=True).items()})
        xm = W / 2
        th = (np.polyval(deep_line, xm) - np.polyval(sup_line, xm)) * math.cos(math.radians(0.5 * (deep_ang + sup_ang)))
        out["v2b_fl_chord_mid"] = float(th / px_per_mm / math.sin(math.radians(max(ax_ + 0.5 * bx_, 3))))
    return out


def streamline_fl(fasc_prob: np.ndarray, apo: dict, px_per_mm: float, a: float, b: float, sign: float,
                  seeds=(0.3, 0.4, 0.5, 0.6, 0.7), axis_rel: bool = False) -> dict:
    """FL along traced fascicle paths (the raters' segmented line), extrapolated straight outside the frame.

    From seeds on the deep aponeurosis (central x) a path is stepped towards the superficial aponeurosis. The
    step direction blends the local fascicle orientation (structure tensor of the fascicle map, weighted by its
    coherence and probability) with the depth-fitted angle model a + b d; outside the image the last direction is
    kept (linear extrapolation). FL = median path length over seeds.
    """
    H, W = fasc_prob.shape
    s, d = apo["sup"], apo["deep"]
    sup_line = s["bot"] if s["bot"] is not None else s["cen"]
    deep_line = d["top"] if d["top"] is not None else d["cen"]
    deep_ang = math.degrees(math.atan(deep_line[0]))
    sup_ang = math.degrees(math.atan(sup_line[0]))
    f = fasc_prob.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, 1.5 * px_per_mm)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    theta = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    wf = coh * cv2.GaussianBlur(f, (0, 0), sig)
    rel_field = ((theta - deep_ang + 90) % 180) - 90  # relative to the deep aponeurosis (axis_rel: corrected below)
    wmax = float(np.percentile(wf, 99)) + 1e-6
    lens = []
    for fr in seeds:
        x, y = fr * W, np.polyval(deep_line, fr * W) - 1.0
        length, n = 0.0, 0
        dvec = None
        while n < 6 * (H + W):
            n += 1
            ys, yd = np.polyval(sup_line, x), np.polyval(deep_line, x)
            if y <= ys:
                break
            dpt = float(np.clip((y - ys) / max(yd - ys, 1.0), 0, 1))
            phi = a + b * dpt
            base = (1 - dpt) * sup_ang + dpt * deep_ang if axis_rel else deep_ang
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                rf = rel_field[yi, xi] - (base - deep_ang)
                ok = np.sign(rf) == sign and 2 <= abs(rf) <= 60 and abs(abs(rf) - phi) < 8.0
                w = min(wf[yi, xi] / wmax, 1.0) if ok else 0.0
                phi = (w * abs(rf) + 0.5 * phi) / (w + 0.5)
                ang = math.radians(base + sign * phi)
                dvec = np.array([math.cos(ang), math.sin(ang)])
                if dvec[1] > 0:
                    dvec = -dvec
            elif dvec is None:
                break
            x, y = x + 2 * dvec[0], y + 2 * dvec[1]  # 2 px steps
            length += 2.0
        if 0 < length and n < 6 * (H + W):
            lens.append(length / px_per_mm)
    return {"v2_fl_stream": float(np.median(lens))} if lens else {}


def mt_variants(apo: dict, px_per_mm: float) -> dict:
    """Vertical inner-edge MT on the raw (unfitted) aponeurosis edges at left/middle/right positions.

    The host draws three vertical lines between the aponeuroses; on straight-line fits every symmetric position
    triple gives the same mean, so the raw per-column edges (median over +-2 % of the width) are used instead.
    """
    W = apo["W"]
    s, d = apo["sup"], apo["deep"]
    cs = s["bot"] if s["bot"] is not None else s["cen"]
    cd = d["top"] if d["top"] is not None else d["cen"]
    out = {}
    half = max(2, int(0.02 * W))
    for name, fr in (("1090", (0.1, 0.5, 0.9)), ("1684", (1 / 6, 0.5, 5 / 6)), ("2575", (0.25, 0.5, 0.75)),
                     ("mid", (0.5,))):
        vals = []
        for f in fr:
            x0 = int(f * W)
            sl = slice(max(0, x0 - half), min(W, x0 + half + 1))
            yb, yt = np.nanmedian(s["bot_raw"][sl]), np.nanmedian(d["top_raw"][sl])
            if not np.isfinite(yb):
                yb = np.polyval(cs, x0)
            if not np.isfinite(yt):
                yt = np.polyval(cd, x0)
            vals.append(yt - yb)
        out[f"v2_mt_raw_{name}"] = float(np.mean(vals) / px_per_mm)
    return out


def analyse(apo_prob: np.ndarray, fasc_prob: np.ndarray, px_per_mm: float) -> dict:
    ap = apo_prob.astype(np.float32) / 255.0 if apo_prob.dtype == np.uint8 else apo_prob
    fp = fasc_prob.astype(np.float32) / 255.0 if fasc_prob.dtype == np.uint8 else fasc_prob
    apo = find_aponeuroses(ap, px_per_mm)
    if apo is None:
        apo = find_aponeuroses(ap, px_per_mm, thr=0.3)
    if apo is None:
        return dict(ok=0)
    out = dict(ok=1, n_bands=apo["n_bands"])
    out.update(muscle_thickness(apo, px_per_mm))
    out.update(fascicles(fp, apo, px_per_mm))
    out["pa_orient"] = orientation_pa(fp, apo, px_per_mm)
    out.update(fascicles_v2(fp, apo, px_per_mm))
    out.update(mt_variants(apo, px_per_mm))
    return out
