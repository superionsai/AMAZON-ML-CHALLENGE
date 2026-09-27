"""Candidate generation, run independently per country (true matches never cross countries).

Two complementary retrieval channels, unioned:
  1. Joint TF-IDF kNN: name skeleton char 2-3-grams + normalised address words,
     top-k S1 neighbours for every S2/S3 record (direction matches the one-to-one
     structure: every S2/S3 record belongs to at most one S1 entity).
  2. Deterministic keys ("tags"): exact equality on rarity-ranked tokens
     (skeleton, house number + rarest street token, two rarest street tokens, ...).
     Ties in rarity are broken alphabetically, so keys are fully reproducible.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import diags, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from normalize import DOMAIN_RE, DBA_RE, norm_addr, norm_name, skel, skel_token

KEYS = ["k_skel", "k_hn_st", "k_st2", "k_nm_st", "k_nm_hn", "k_post_nm"]


def prep(df: pd.DataFrame, keep_raw: bool = False) -> pd.DataFrame:
    """Per-record normalised fields. Input columns: entity_id, business_name, business_address."""
    names = df.business_name.fillna("").values
    addrs = df.business_address.fillna("").values
    nn = [norm_name(x) for x in names]
    na = [norm_addr(x) for x in addrs]
    at = [a.split() for a in na]
    P = pd.DataFrame({"id": df.entity_id.values, "nname": nn, "naddr": na})
    P["skel"] = [skel(x) for x in nn]
    P["ntok"] = [[y for y in (skel_token(t) for t in x.split()) if len(y) >= 2] for x in nn]
    P["hn"] = [next((t for t in ts if t.isdigit()), "") for ts in at]
    P["nums"] = [" ".join(sorted({t for t in ts if t.isdigit()})) for ts in at]
    P["atok"] = [[t for t in ts if not t.isdigit() and len(t) >= 3] for ts in at]
    P["postal"] = [next((t for t in reversed(ts) if t.isdigit() and len(t) in (5, 6)), "") for ts in at]
    P["nonascii"] = [not x.isascii() for x in names]
    P["domain"] = [bool(DOMAIN_RE.search(x)) for x in names]
    P["dba"] = [bool(DBA_RE.search(x)) for x in names]
    P["noaddr"] = [a == "" for a in na]
    if keep_raw:   # original text for the cross-encoder
        P["raw"] = [f"{n.strip()} ; {a.strip()}" for n, a in zip(names, addrs)]
    return P


def _two_rarest(tokens, df):
    s = sorted(set(tokens), key=lambda t: (df[t], t))
    return (s + ["", ""])[:2]


def add_rarity_and_keys(P1: pd.DataFrame, PQ: pd.DataFrame) -> None:
    """Rarity is measured on S1+S2+S3 of the country together (unsupervised, test-safe)."""
    dfn = Counter(t for L in pd.concat([P1.ntok, PQ.ntok]) for t in set(L))
    dfa = Counter(t for L in pd.concat([P1.atok, PQ.atok]) for t in set(L))
    for P in (P1, PQ):
        rn = [_two_rarest(L, dfn) for L in P.ntok]
        ra = [_two_rarest(L, dfa) for L in P.atok]
        P["rn1"] = [x[0] for x in rn]
        P["ra1"] = [x[0] for x in ra]
        P["ra2"] = [x[1] for x in ra]
        both = lambda a, b: (a != "") & (b != "")
        P["k_skel"] = P.skel.where(P.skel.str.len() >= 4, "")
        P["k_hn_st"] = (P.hn + "|" + P.ra1).where(both(P.hn, P.ra1), "")
        P["k_st2"] = [("|".join(sorted((a, b))) if a and b else "") for a, b in zip(P.ra1, P.ra2)]
        P["k_nm_st"] = (P.rn1 + "|" + P.ra1).where(both(P.rn1, P.ra1), "")
        P["k_nm_hn"] = (P.rn1 + "|" + P.hn).where(both(P.rn1, P.hn), "")
        P["k_post_nm"] = (P.postal + "|" + P.skel.str[:3]).where(both(P.postal, P.skel), "")


def _joint(N, A, w):
    return normalize(hstack([normalize(N) * w, normalize(A) * (1 - w)]).tocsr())


class CountryBlocker:
    def __init__(self, P1: pd.DataFrame, PQ: pd.DataFrame, cfg):
        self.P1, self.PQ, self.cfg = P1, PQ, cfg
        vn = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2,
                             sublinear_tf=True, dtype=np.float32)
        va = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True, dtype=np.float32)
        vn.fit(pd.concat([P1.skel, PQ.skel]))          # vocab from S1+S2+S3: rare streets survive
        va.fit(pd.concat([P1.naddr, PQ.naddr]))
        self.N1, self.NQ = normalize(vn.transform(P1.skel)), normalize(vn.transform(PQ.skel))
        self.A1, self.AQ = normalize(va.transform(P1.naddr)), normalize(va.transform(PQ.naddr))
        self.X1, self.XQ = _joint(self.N1, self.A1, cfg.w_name), _joint(self.NQ, self.AQ, cfg.w_name)
        # retrieval-only copy without very common features: keeps sparse matmul sparse.
        # Pair features (cos_all etc.) still use the full vectors.
        df1 = np.bincount(self.X1.indices, minlength=self.X1.shape[1])
        keep = diags((df1 <= max(cfg.knn_max_df * self.X1.shape[0], 20)).astype(np.float32))
        self.R1T = normalize(self.X1 @ keep).T.tocsr()
        self.RQ = normalize(self.XQ @ keep).tocsr()
        # key blocks: size caps are evaluated on the WHOLE country, so chunking never changes output
        self.key_s1 = {}
        for k in KEYS:
            c1, cq = P1[k].value_counts(), PQ[k].value_counts()
            ok = set(c1.index[c1 <= cfg.key_cap_s1]) & set(cq.index[cq <= cfg.key_cap_q])
            ok.discard("")
            d1 = pd.DataFrame({"key": P1[k].values, "s": np.arange(len(P1), dtype=np.int32)})
            self.key_s1[k] = (d1[d1.key.isin(ok)], ok)

    def _knn(self, lo, hi):
        M = sp_matmul_topn(self.RQ[lo:hi], self.R1T, top_n=self.cfg.knn_k, threshold=1e-6,
                           n_threads=self.cfg.n_threads).tocoo()
        return (M.row + lo).astype(np.int32), M.col.astype(np.int32)

    def pairs(self, lo: int, hi: int) -> pd.DataFrame:
        q, s = self._knn(lo, hi)
        frames = [pd.DataFrame({"q": q, "s": s, "in_knn": np.uint8(1)})]
        for k in KEYS:
            d1, ok = self.key_s1[k]
            d2 = pd.DataFrame({"key": self.PQ[k].values[lo:hi], "q": np.arange(lo, hi, dtype=np.int32)})
            m = d1.merge(d2[d2.key.isin(ok)], on="key")
            frames.append(pd.DataFrame({"q": m.q.values, "s": m.s.values, k: np.uint8(1)}))
        flags = ["in_knn"] + KEYS
        C = pd.concat(frames, ignore_index=True)
        C[flags] = C[flags].fillna(0).astype(np.uint8)
        C = C.groupby(["q", "s"], sort=True, as_index=False)[flags].max()
        s, q = C.s.values, C.q.values
        C["cos_all"] = pair_dot(self.X1, self.XQ, s, q)
        C["cos_name"] = pair_dot(self.N1, self.NQ, s, q)
        C["cos_addr"] = pair_dot(self.A1, self.AQ, s, q)
        return C

    def chunks(self):
        n, step = len(self.PQ), self.cfg.query_chunk
        for lo in range(0, n, step):
            yield lo, min(lo + step, n)


def pair_dot(M1, M2, s, q, chunk=2_000_000):
    out = np.empty(len(s), dtype=np.float32)
    for i in range(0, len(s), chunk):
        j = slice(i, i + chunk)
        out[j] = np.asarray(M1[s[j]].multiply(M2[q[j]]).sum(axis=1)).ravel()
    return out
