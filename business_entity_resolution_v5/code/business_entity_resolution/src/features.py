"""Pairwise features. Country is never a feature (test has an unseen country)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from blocking import KEYS


def group_ctx(key, val):
    """Per-group (rank 1=best, group size, margin vs best OTHER member). Deterministic, O(n log n)."""
    key = np.asarray(key)
    val = np.asarray(val, dtype=np.float32)
    n = len(key)
    if n == 0:
        z = np.zeros(0, np.float32)
        return z, z, z
    order = np.lexsort((-val, key))
    k, v = key[order], val[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    size = np.diff(np.r_[starts, n])
    rank = np.arange(n) - starts[gid] + 1
    top1 = v[starts]
    top2 = np.where(size > 1, v[np.minimum(starts + 1, n - 1)], 0.0)
    margin = v - np.where(rank == 1, top2[gid], top1[gid])
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return (rank[inv].astype(np.float32), size[gid][inv].astype(np.float32),
            margin[inv].astype(np.float32))


def _share(key, val):
    codes, _ = pd.factorize(key)
    tot = np.bincount(codes, weights=val)
    return (val / np.maximum(tot[codes], 1e-9)).astype(np.float32)


# ------------------------------------------------------------------ stage 1 (cheap)
S1_FEATS = (["cos_all", "cos_name", "cos_addr", "in_knn"] + KEYS +
            ["skel_eq", "hn_both", "hn_eq", "post_eq", "rn1_eq", "ra1_eq", "ra_cross",
             "noaddr_a", "noaddr_b", "nonascii_a", "nonascii_b", "domain_b", "dba_b",
             "q_rank", "q_n", "q_margin", "qn_margin", "qa_margin", "q_share"])


def stage1_features(C: pd.DataFrame, P1: pd.DataFrame, PQ: pd.DataFrame) -> pd.DataFrame:
    s, q = C.s.values, C.q.values
    A = lambda c: P1[c].values[s]
    B = lambda c: PQ[c].values[q]
    ne = lambda x: x != ""
    C["skel_eq"] = ne(A("skel")) & (A("skel") == B("skel"))
    C["hn_both"] = ne(A("hn")) & ne(B("hn"))
    C["hn_eq"] = C.hn_both.values & (A("hn") == B("hn"))
    C["post_eq"] = ne(A("postal")) & (A("postal") == B("postal"))
    C["rn1_eq"] = ne(A("rn1")) & (A("rn1") == B("rn1"))
    C["ra1_eq"] = ne(A("ra1")) & (A("ra1") == B("ra1"))
    C["ra_cross"] = (ne(A("ra1")) & ((A("ra1") == B("ra2")) | (A("ra2") == B("ra1"))))
    for c, src in [("noaddr_a", A), ("nonascii_a", A)]:
        C[c] = src(c[:-2])
    for c in ["noaddr_b", "nonascii_b", "domain_b", "dba_b"]:
        C[c] = B(c[:-2])
    C["q_rank"], C["q_n"], C["q_margin"] = group_ctx(q, C.cos_all.values)
    C["qn_margin"] = group_ctx(q, C.cos_name.values)[2]
    C["qa_margin"] = group_ctx(q, C.cos_addr.values)[2]
    C["q_share"] = _share(q, C.cos_all.values.astype(np.float64))
    if "emb_cos" in C:
        C["qe_margin"] = group_ctx(q, C.emb_cos.values)[2]
    bools = [c for c in S1_FEATS if C[c].dtype == bool]
    C[bools] = C[bools].astype(np.uint8)
    return C


# ------------------------------------------------------------------ stage 2 (full)
STR_FEATS = ["jw_name", "ratio_name", "tset_name", "tsort_name", "partial_name", "jw_skel",
             "lev_skel", "tset_addr", "ratio_addr", "partial_addr", "hn_lev", "hn_suffix",
             "num_jac", "tok_jac", "len_a", "len_b"]
CTX_FEATS = ["p1", "p1_q_rank", "p1_q_margin", "p1_q_share", "p1_s_rank", "p1_s_n",
             "p1_s_margin", "s_rank", "s_n", "s_margin", "skel_support", "hn_support", "s_hi"]
S2_FEATS = S1_FEATS + STR_FEATS + CTX_FEATS
EMB1 = ["emb_cos", "qe_margin"]
EMB2 = ["emb_q_rank", "se_margin"]


def feature_lists(use_emb: bool):
    s1 = S1_FEATS + (EMB1 if use_emb else [])
    s2 = s1 + STR_FEATS + CTX_FEATS + (EMB2 if use_emb else [])
    return s1, s2


def _rf(a, b, scorer, scale=1.0):
    return cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / scale


def _jac(xs, ys):
    out = np.zeros(len(xs), dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x and y:
            a, b = set(x.split()), set(y.split())
            out[i] = len(a & b) / len(a | b)
    return out


def string_features(C: pd.DataFrame, P1: pd.DataFrame, PQ: pd.DataFrame) -> pd.DataFrame:
    s, q = C.s.values, C.q.values
    na, nb = P1.nname.values[s], PQ.nname.values[q]
    ka, kb = P1.skel.values[s], PQ.skel.values[q]
    aa, ab = P1.naddr.values[s], PQ.naddr.values[q]
    ha, hb = P1.hn.values[s], PQ.hn.values[q]
    C["jw_name"] = _rf(na, nb, JaroWinkler.normalized_similarity)
    C["ratio_name"] = _rf(na, nb, fuzz.ratio, 100)
    C["tset_name"] = _rf(na, nb, fuzz.token_set_ratio, 100)
    C["tsort_name"] = _rf(na, nb, fuzz.token_sort_ratio, 100)
    C["partial_name"] = _rf(na, nb, fuzz.partial_ratio, 100)
    C["jw_skel"] = _rf(ka, kb, JaroWinkler.normalized_similarity)
    C["lev_skel"] = _rf(ka, kb, Levenshtein.normalized_similarity)
    C["tset_addr"] = _rf(aa, ab, fuzz.token_set_ratio, 100)
    C["ratio_addr"] = _rf(aa, ab, fuzz.ratio, 100)
    C["partial_addr"] = _rf(aa, ab, fuzz.partial_ratio, 100)
    both = (ha != "") & (hb != "")
    C["hn_lev"] = np.where(both, _rf(ha, hb, Levenshtein.distance), -1).astype(np.float32)
    C["hn_suffix"] = np.array([bool(x and y and x != y and (x.endswith(y) or y.endswith(x)))
                               for x, y in zip(ha, hb)], dtype=np.uint8)
    C["num_jac"] = _jac(P1.nums.values[s], PQ.nums.values[q])
    C["tok_jac"] = _jac(na, nb)
    C["len_a"] = np.fromiter((len(x) for x in na), np.float32, len(na))
    C["len_b"] = np.fromiter((len(x) for x in nb), np.float32, len(nb))
    return C


def context_features(C: pd.DataFrame, PQ: pd.DataFrame) -> pd.DataFrame:
    """Needs ALL pruned rows of a country at once (S1-side groups span query chunks)."""
    s, q, p1 = C.s.values, C.q.values, C.p1.values.astype(np.float32)
    C["p1_q_rank"], _, C["p1_q_margin"] = group_ctx(q, p1)
    C["p1_q_share"] = _share(q, p1.astype(np.float64))
    C["p1_s_rank"], C["p1_s_n"], C["p1_s_margin"] = group_ctx(s, p1)
    C["s_rank"], C["s_n"], C["s_margin"] = group_ctx(s, C.cos_all.values)
    if "emb_cos" in C:
        C["emb_q_rank"] = group_ctx(q, C.emb_cos.values)[0]
        C["se_margin"] = group_ctx(s, C.emb_cos.values)[2]
    hi = (p1 > 0.5).astype(np.int32)
    tmp = pd.DataFrame({"s": s, "k": PQ.skel.values[q], "h": PQ.hn.values[q], "hi": hi})
    C["skel_support"] = (tmp.groupby(["s", "k"]).hi.transform("sum").values - hi).astype(np.float32)
    C["hn_support"] = ((tmp.groupby(["s", "h"]).hi.transform("sum").values - hi)
                       * (tmp.h.values != "")).astype(np.float32)
    C["s_hi"] = (tmp.groupby("s").hi.transform("sum").values - hi).astype(np.float32)
    return C
