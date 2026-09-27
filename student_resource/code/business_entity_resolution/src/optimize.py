"""Metric-driven optimisers and decoders for V6.

Upgrades:
* prune_threshold + stage1_safety_mask:
    Exact closed-form threshold + structural safety net ensuring top query neighbors,
    exact skeletons, and strong anchors are never dropped.
* select_matches:
    One-to-one assignment + per-S1 expected-F0.5-optimal subset with dedicated singleton
    preservation gate (protects against singleton false-positive penalties).
* coord_ascent:
    Derivative-free calibration of (T, b, c, tau_min, p_empty_mult) on the exact macro F0.5.
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


# --------------------------------------------------------------- stage-1 pruning & safety net
def prune_threshold(p1, y, target=0.999):
    pos = np.sort(np.asarray(p1)[np.asarray(y) == 1])
    if len(pos) == 0:
        return 0.0
    idx = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[min(idx, len(pos) - 1)])


def stage1_safety_mask(C, p1, tau):
    """Safety net: never prune high-confidence structural anchors or top query neighbors (IDEA.md Section 18)."""
    keep = (p1 >= tau)
    if "q_rank" in C:
        keep |= (C.q_rank.values == 1)   # Top-1 candidate for query
    if "skel_eq" in C and "cos_name" in C:
        keep |= (C.skel_eq.values == 1) & (C.cos_name.values >= 0.50)
    if "k_skel" in C:
        keep |= (C.k_skel.values == 1)
    if "k_name_sort" in C:
        keep |= (C.k_name_sort.values == 1)
    if "k_nm_hn" in C:
        keep |= (C.k_nm_hn.values == 1)
    if "hn_eq" in C and "ra1_eq" in C:
        keep |= (C.hn_eq.values == 1) & (C.ra1_eq.values == 1)
    if "cos_all" in C:
        keep |= (C.cos_all.values >= 0.70)
    return keep


# --------------------------------------------------------------- decision rule & singleton gate
def select_matches(s, q, logit, T=1.0, b=0.0, c=0.0, tau_min=0.40, p_empty_mult=1.05):
    """One-to-one query exclusivity + expected-F0.5 set decoder with singleton gate."""
    s, q = np.asarray(s, np.int64), np.asarray(q, np.int64)
    if len(s) == 0:
        return s, q
    p = 1.0 / (1.0 + np.exp(-(np.asarray(logit, np.float64) / T + b)))

    # Hard noise filter: discard extreme low-confidence candidates (under 0.20)
    valid = p >= 0.20
    s, q, p = s[valid], q[valid], p[valid]
    if len(s) == 0:
        return s, q

    # One-to-one exclusivity: each S2/S3 query record keeps only its best S1 entity
    o = np.lexsort((s, -p, q))
    first = np.r_[True, q[o][1:] != q[o][:-1]]
    k = o[first]
    s, q, p = s[k], q[k], p[k]

    # Group by S1 entity and sort candidates by probability descending
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

    # Empty probability calculation
    p_empty = np.exp(np.bincount(gid, weights=np.log(np.clip(1.0 - p, 1e-12, 1.0))))

    # Singleton gate (IDEA.md Section 31):
    # Top-1 candidate probability for each entity
    max_p = p[starts]
    has_strong_candidate = max_p >= tau_min

    # Entity is kept non-empty only if it beats p_empty and passes singleton gate
    entity_active = (p_empty * p_empty_mult < best) & has_strong_candidate
    keep = (j <= best_j[gid]) & entity_active[gid]
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
