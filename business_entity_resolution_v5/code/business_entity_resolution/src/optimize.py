"""Metric-driven optimisers. Nothing here is a hand-picked threshold.

* prune_threshold   - exact closed-form: the highest stage-1 cut that keeps a target share
                      of blocked true pairs (minimises candidate count under a recall floor).
* select_matches    - one-to-one assignment + per-S1 expected-F0.5-optimal subset.
                      For each S1 entity, candidates are sorted by calibrated probability and
                      the prefix length j maximising
                          E[F0.5](j) ~= 1.25 * sum_{i<=j} p_i / (j + 0.25 * (sum_all p + c))
                      is chosen; the empty set is chosen when P(no match) = prod(1 - p_i)
                      beats it (this is what earns the 1.0 on singletons).
* coord_ascent      - derivative-free maximiser of the EXACT macro F0.5 on validation over the
                      calibration (T, b) and missed-mass prior c: coarse grid to bracket the
                      optimum, then golden-section refinement, per coordinate, repeated sweeps.
"""
import numpy as np


# --------------------------------------------------------------- metric
def macro_f05(pred_s, pred_q, true_s, true_q, n_s):
    """Exact challenge metric. s codes in [0, n_s) cover EVERY evaluated S1 entity."""
    pred_s, pred_q = np.asarray(pred_s, np.int64), np.asarray(pred_q, np.int64)
    true_s, true_q = np.asarray(true_s, np.int64), np.asarray(true_q, np.int64)
    base = max(pred_q.max(initial=0), true_q.max(initial=0)) + 1
    hit = np.isin(pred_s * base + pred_q, true_s * base + true_q)
    n_pred = np.bincount(pred_s, minlength=n_s)
    n_true = np.bincount(true_s, minlength=n_s)
    tp = np.bincount(pred_s[hit], minlength=n_s)
    P = tp / np.maximum(n_pred, 1)
    R = tp / np.maximum(n_true, 1)
    f = np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    f = np.where((n_pred == 0) & (n_true == 0), 1.0, f)
    return float(f.mean())


# --------------------------------------------------------------- stage-1 pruning
def prune_threshold(p1, y, target):
    pos = np.sort(np.asarray(p1)[np.asarray(y) == 1])
    if len(pos) == 0:
        return 0.0
    idx = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[min(idx, len(pos) - 1)])


# --------------------------------------------------------------- decision rule
def select_matches(s, q, logit, T=1.0, b=0.0, c=0.0):
    s, q = np.asarray(s, np.int64), np.asarray(q, np.int64)
    if len(s) == 0:
        return s, q
    p = 1.0 / (1.0 + np.exp(-(np.asarray(logit, np.float64) / T + b)))
    # one-to-one: each S2/S3 record keeps only its best S1 (ties -> smaller s, deterministic)
    o = np.lexsort((s, -p, q))
    first = np.r_[True, q[o][1:] != q[o][:-1]]
    k = o[first]
    s, q, p = s[k], q[k], p[k]
    # per-S1 expected-F0.5-optimal prefix
    o = np.lexsort((q, -p, s))
    s, q, p = s[o], q[o], p[o]
    start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    j = np.arange(len(s)) - starts[gid] + 1
    cs = np.cumsum(p)
    cs_g = cs - np.r_[0.0, cs][starts][gid]
    tot = np.bincount(gid, weights=p)
    ef = 1.25 * cs_g / (j + 0.25 * (tot[gid] + c))
    best = np.maximum.reduceat(ef, starts)
    best_j = np.minimum.reduceat(np.where(ef >= best[gid], j, 10 ** 9), starts)
    p_empty = np.exp(np.bincount(gid, weights=np.log(np.clip(1.0 - p, 1e-12, 1.0))))
    keep = (j <= best_j[gid]) & (p_empty[gid] < best[gid])
    return s[keep], q[keep]


# --------------------------------------------------------------- optimisers
def golden(f, lo, hi, iters=18):
    g = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (c, fc) if fc >= fd else (d, fd)


def coord_ascent(f, x0, bounds, sweeps=3, grid=9, iters=18, log=print):
    x = dict(x0)
    best = f(x)
    log(f"    start {x} -> {best:.5f}")
    for sw in range(sweeps):
        improved = False
        for k, (lo, hi) in bounds.items():
            pts = np.linspace(lo, hi, grid)
            vals = [f({**x, k: v}) for v in pts]
            i = int(np.argmax(vals))
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, grid - 1)]
            v, fv = golden(lambda t: f({**x, k: t}), a, b, iters)
            fv_best, v_best = max((vals[i], pts[i]), (fv, v))
            if fv_best > best + 1e-7:
                x[k], best, improved = float(v_best), fv_best, True
        log(f"    sweep {sw + 1}: {x} -> {best:.5f}")
        if not improved:
            break
    return x, best
