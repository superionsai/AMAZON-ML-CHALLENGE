"""Business Entity Resolution V6 - end-to-end pipeline.

Implements all IDEA.md guidelines to target >=0.99 macro F0.5:
  1. Multi-channel candidate generation: Joint TF-IDF kNN U Name Char kNN U Addr Word kNN U 10 Rarity Keys
  2. Stage 1 LightGBM + Pruning Safety Net (protects top query neighbors and structural keys)
     -> Surviving candidate pool forms output/candidate_pairs.tsv
  3. Stage 2 LightGBM with ~65 rich features (numeric contradiction, cross-field interactions, prefix/sort/acronym, context ranks/margins)
  4. Metric-aware decision rule: 1-to-1 query assignment, per-S1 expected-F0.5 subset decoding,
     and dedicated singleton existence gate tuned on exact validation macro F0.5
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import CountryBlocker, add_rarity_and_keys, prep  # noqa: E402
from config import Config  # noqa: E402
from features import context_features, feature_lists, stage1_features, string_features  # noqa: E402
from io_utils import load_split, truth_pairs, write_id_lists  # noqa: E402
from optimize import coord_ascent, macro_f05, prune_threshold, select_matches, stage1_safety_mask  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def lgb_params(cfg):
    return dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                seed=cfg.seed, deterministic=True, force_row_wise=True, verbose=-1,
                num_threads=cfg.n_threads)


def fit_lgb(cfg, Xtr, ytr, Xva=None, yva=None, rounds=None):
    dtr = lgb.Dataset(Xtr, ytr)
    if rounds is not None:
        return lgb.train(lgb_params(cfg), dtr, rounds)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(lgb_params(cfg), dtr, 3000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])


def F(df, cols):
    return df[cols].values.astype(np.float32)


# ------------------------------------------------------------------ shared building blocks
def build_country(S, Qc, cfg, tag, m1=None, tau=None):
    s1_feats, _ = feature_lists(cfg.use_emb)
    P1, PQ = prep(S, keep_raw=cfg.use_ce), prep(Qc, keep_raw=cfg.use_ce)
    add_rarity_and_keys(P1, PQ)
    blk = CountryBlocker(P1, PQ, cfg)
    E1 = EQ = None
    if cfg.use_emb:
        import neural
        E1 = neural.NameEmbeddings(S.business_name.values, cfg, f"{tag}_s1", log=log)
        EQ = neural.NameEmbeddings(Qc.business_name.values, cfg, f"{tag}_q", log=log)
    parts, n_raw = [], 0
    log(f"  prepared + indexed: S1={len(P1):,} queries={len(PQ):,}")
    for lo, hi in blk.chunks():
        C = blk.pairs(lo, hi)
        if E1 is not None:
            C["emb_cos"] = neural.emb_cos(E1, EQ, C.s.values, C.q.values)
        C = stage1_features(C, P1, PQ)
        n_raw += len(C)
        log(f"  queries {hi:,}/{len(PQ):,}: {len(C):,} pairs in chunk")
        if m1 is not None:
            C["p1"] = m1.predict(F(C, s1_feats))
            if cfg.safety_net:
                keep = stage1_safety_mask(C, C.p1.values, tau)
            else:
                keep = C.p1.values >= tau
            C = C[keep]
        parts.append(C)
    C = pd.concat(parts, ignore_index=True)
    return P1, PQ, C, n_raw


def stage2_frame(C, P1, PQ, step=2_000_000):
    C = context_features(C.reset_index(drop=True), PQ)
    parts = [string_features(C.iloc[i:i + step].copy(), P1, PQ) for i in range(0, len(C), step)]
    return pd.concat(parts, ignore_index=True) if parts else C


# ------------------------------------------------------------------ train
def pair_texts_multi(Dsub, recs):
    ta = np.empty(len(Dsub), dtype=object)
    tb = np.empty(len(Dsub), dtype=object)
    for country, (P1, PQ) in recs.items():
        m = Dsub.country.values == country
        if m.any():
            ta[m] = P1.raw.values[Dsub.s.values[m]]
            tb[m] = PQ.raw.values[Dsub.q.values[m]]
    return ta.tolist(), tb.tolist()


def train(cfg):
    S1_FEATS, S2_FEATS = feature_lists(cfg.use_emb)
    s1, Q, gt = load_split(cfg.data_dir, "train")
    tp = truth_pairs(gt)
    q2s = dict(zip(tp.q_id, tp.s1_id))
    matched = set(tp.q_id)
    log(f"loaded train: S1={len(s1):,}  S2+S3={len(Q):,}  true pairs={len(tp):,}")

    recs, Cs, val_s1 = {}, [], []
    for country in sorted(s1.country.unique()):
        Sall = s1[s1.country == country]
        E = Sall.sample(frac=cfg.train_s1_frac, random_state=cfg.seed)
        eset = set(E.entity_id)
        mq = set(tp.q_id[tp.s1_id.isin(eset)])
        Qc = Q[Q.country == country]
        dis = Qc[~Qc.entity_id.isin(matched)].sample(frac=cfg.train_s1_frac, random_state=cfg.seed)
        Qs = pd.concat([Qc[Qc.entity_id.isin(mq)], dis], ignore_index=True)
        P1, PQ, C, _ = build_country(Sall, Qs, cfg, f"train{cfg.train_s1_frac}_{country}")
        C["s1_id"], C["q_id"] = P1.id.values[C.s.values], PQ.id.values[C.q.values]
        C["y"] = (C.q_id.map(q2s).values == C.s1_id.values).astype(np.uint8)
        C["in_e"] = C.s1_id.isin(eset).values
        C["country"] = country
        recs[country] = (P1, PQ)
        Cs.append(C)
        val_s1 += list(E.entity_id.sample(frac=cfg.val_frac, random_state=cfg.seed))
        log(f"{country}: indexed S1={len(Sall):,}  sampled E={len(E):,}  queries={len(Qs):,}  "
            f"union pairs={len(C):,}  per sampled S1={len(C) / len(E):.1f}  "
            f"union recall={C.y.sum() / max(len(mq), 1):.4%}")

    C = pd.concat(Cs, ignore_index=True)
    del Cs
    val_set = set(val_s1)
    is_val = C.s1_id.isin(val_set).values
    is_tr = C.in_e.values & ~is_val
    tv = tp[tp.s1_id.isin(val_set)]
    log(f"val: {len(val_s1):,} S1 entities, {len(tv):,} true pairs; "
        f"blocking recall ceiling={C.y.values[is_val].sum() / len(tv):.4%}")

    # ---------------- stage 1 (+ out-of-fold p1 for training rows)
    X, y = F(C, S1_FEATS), C.y.values
    m1 = fit_lgb(cfg, X[is_tr], y[is_tr], X[is_val], y[is_val])
    r1 = m1.best_iteration or m1.current_iteration()
    p1 = m1.predict(X, num_iteration=r1).astype(np.float32)
    fold = (pd.util.hash_array(C.q_id.values.astype(str)) % 3).astype(int)
    for f in range(3):
        a, b = is_tr & (fold != f), is_tr & (fold == f)
        p1[b] = fit_lgb(cfg, X[a], y[a], rounds=r1).predict(X[b])
    del X
    tau = prune_threshold(p1[is_val], y[is_val], cfg.prune_recall)
    if cfg.safety_net:
        keep = stage1_safety_mask(C, p1, tau)
    else:
        keep = p1 >= tau
    C["p1"] = p1
    log(f"stage1: rounds={r1}  tau={tau:.5f}  val candidates per S1: "
        f"{(is_val & keep).sum() / len(val_s1):.2f} (from {is_val.sum() / len(val_s1):.2f})  "
        f"recall after pruning={y[is_val & keep].sum() / len(tv):.4%}")

    # ---------------- stage 2
    D = []
    for country, (P1, PQ) in recs.items():
        Cc = C[keep & (C.country.values == country)]
        D.append(stage2_frame(Cc, P1, PQ))
    D = pd.concat(D, ignore_index=True)
    del C
    dv = D.s1_id.isin(val_set).values
    dtr = D.in_e.values & ~dv
    m2 = fit_lgb(cfg, F(D[dtr], S2_FEATS), D.y.values[dtr], F(D[dv], S2_FEATS), D.y.values[dv])
    r2 = m2.best_iteration or m2.current_iteration()
    V = D[D.q_id.isin(set(D.q_id.values[dv])).values]
    logit = m2.predict(F(V, S2_FEATS), num_iteration=r2, raw_score=True)

    # ---------------- optional cross-encoder on uncertain zone
    ce = np.zeros(len(V), dtype=np.float32)
    ce_dir = os.path.join(cfg.model_dir, "cross_encoder")
    if cfg.use_ce:
        import neural
        T = D[dtr]
        p = T.p1.values.astype(np.float64)
        w = 0.05 + 4.0 * p * (1.0 - p)
        n = min(cfg.ce_train_pairs, len(T))
        pick = np.random.RandomState(cfg.seed).choice(len(T), size=n, replace=False, p=w / w.sum())
        T = T.iloc[np.sort(pick)]
        ta, tb = pair_texts_multi(T, recs)
        log(f"cross-encoder: training on {n:,} pairs ({T.y.mean():.3f} positive)")
        neural.train_cross_encoder(ta, tb, T.y.values, cfg, ce_dir, log=log)
        zone = np.abs(logit) < cfg.ce_margin
        ta, tb = pair_texts_multi(V[zone], recs)
        ce[zone] = neural.ce_score(ta, tb, cfg, ce_dir, log=log)
        log(f"cross-encoder: rescored {zone.sum():,} val pairs ({zone.mean():.3f} of val candidates)")

    # ---------------- decision-rule optimisation on exact metric
    s_cat = pd.Index(val_s1)
    q_cat = pd.Index(pd.unique(np.r_[V.q_id.values, tv.q_id.values]))
    vs, vq = s_cat.get_indexer(V.s1_id), q_cat.get_indexer(V.q_id)
    ts, tq = s_cat.get_indexer(tv.s1_id), q_cat.get_indexer(tv.q_id)
    n_s = len(s_cat)
    other = pd.Index(pd.unique(V.s1_id.values[vs < 0]))
    vs_all = np.where(vs >= 0, vs, n_s + other.get_indexer(V.s1_id.values))
    vs = vs_all
    pos = (V.y.values == 1) & (vs < n_s)
    log(f"val oracle F0.5 (perfect matcher on these candidates) = "
        f"{macro_f05(vs[pos], vq[pos], ts, tq, n_s):.5f}")

    def run_rule(x):
        ps, pq = select_matches(
            vs_all, vq, logit + x.get("w", 0.0) * ce,
            T=x["T"], b=x["b"], c=x["c"],
            tau_min=x.get("tau_min", 0.40),
            p_empty_mult=x.get("p_empty_mult", 1.05)
        )
        m = ps < n_s
        return ps[m], pq[m]

    def objective(x):
        ps, pq = run_rule(x)
        return macro_f05(ps, pq, ts, tq, n_s)

    x0 = {"T": 1.0, "b": 0.0, "c": 0.0, "tau_min": 0.45, "p_empty_mult": 1.05}
    bounds = {
        "T": (0.3, 4.0),
        "b": (-3.0, 3.0),
        "c": (0.0, 2.5),
        "tau_min": (0.30, 0.70),
        "p_empty_mult": (0.80, 1.40)
    }
    if cfg.use_ce:
        x0["w"], bounds["w"] = 0.0, (0.0, 3.0)
        log(f"val F0.5 without cross-encoder = {objective(x0):.5f}")
    x, best = coord_ascent(objective, x0, bounds, log=log)
    log(f"val F0.5 optimised (1-1 + expected-F rule + singleton gate) = {best:.5f}   params={x}")
    ps, _ = run_rule(x)
    n_pred = np.bincount(ps, minlength=n_s)
    n_true = np.bincount(ts, minlength=n_s)
    log(f"val sanity: matches per S1 pred={n_pred.mean():.2f} true={n_true.mean():.2f} | "
        f"singleton rate pred={(n_pred == 0).mean():.3f} true={(n_true == 0).mean():.3f}")

    os.makedirs(cfg.model_dir, exist_ok=True)
    for f in os.listdir(cfg.model_dir):
        if f.startswith("pred_") and f.endswith(".pkl"):
            os.remove(os.path.join(cfg.model_dir, f))
    m1.save_model(os.path.join(cfg.model_dir, "stage1.txt"), num_iteration=r1)
    m2.save_model(os.path.join(cfg.model_dir, "stage2.txt"), num_iteration=r2)
    imp = pd.Series(m2.feature_importance("gain"), index=S2_FEATS).sort_values(ascending=False)
    log("top stage-2 features:\n" + imp.head(15).to_string())
    with open(os.path.join(cfg.model_dir, "params.json"), "w") as f:
        json.dump({"tau": tau, "decision": x, "val_f05": best, "s1_feats": S1_FEATS,
                   "s2_feats": S2_FEATS, "use_emb": cfg.use_emb, "use_ce": cfg.use_ce,
                   "config": cfg.dict()}, f, indent=2)


# ------------------------------------------------------------------ predict
def predict(cfg):
    m1 = lgb.Booster(model_file=os.path.join(cfg.model_dir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(cfg.model_dir, "stage2.txt"))
    with open(os.path.join(cfg.model_dir, "params.json")) as f:
        params = json.load(f)
    tau, x = params["tau"], params["decision"]
    cfg.use_emb, cfg.use_ce = params.get("use_emb", False), params.get("use_ce", False)
    for k in ("knn_k", "knn_k_name", "knn_k_addr", "w_name", "knn_max_df", "key_cap_s1",
              "key_cap_q", "nn_model", "ce_margin", "ce_max_len", "prune_recall", "safety_net", "singleton_gate"):
        if k in params.get("config", {}):
            setattr(cfg, k, params["config"][k])
    log(f"restored training settings: knn_k={cfg.knn_k} key caps={cfg.key_cap_s1}/{cfg.key_cap_q} "
        f"max_df={cfg.knn_max_df} safety_net={cfg.safety_net} emb={cfg.use_emb} ce={cfg.use_ce}")
    S2_FEATS = params.get("s2_feats") or feature_lists(cfg.use_emb)[1]
    ce_dir = os.path.join(cfg.model_dir, "cross_encoder")
    s1, Q, _ = load_split(cfg.data_dir, "test")
    log(f"loaded test: S1={len(s1):,}  S2+S3={len(Q):,}  countries={sorted(s1.country.unique())}")

    out = []
    for country in sorted(s1.country.unique()):
        cache = os.path.join(cfg.model_dir, f"pred_{country}.pkl")
        if os.path.exists(cache):
            out.append(pd.read_pickle(cache))
            log(f"{country}: loaded cached predictions ({cache})")
            continue
        S, Qc = s1[s1.country == country], Q[Q.country == country]
        if len(S) == 0 or len(Qc) == 0:
            log(f"{country}: no S2/S3 records -> all singletons")
            continue
        log(f"{country}: start")
        P1, PQ, C, n_raw = build_country(S, Qc, cfg, f"test_{country}", m1, tau)
        C = stage2_frame(C, P1, PQ)
        C["logit"] = m2.predict(F(C, S2_FEATS), raw_score=True)
        ce = np.zeros(len(C), dtype=np.float32)
        if cfg.use_ce:
            import neural
            u = np.abs(C.logit.values)
            idx = np.flatnonzero(u < cfg.ce_margin)
            if len(idx) > cfg.ce_max_rows:
                idx = idx[np.argsort(u[idx], kind="stable")[:cfg.ce_max_rows]]
            ta, tb = P1.raw.values[C.s.values[idx]].tolist(), PQ.raw.values[C.q.values[idx]].tolist()
            ce[idx] = neural.ce_score(ta, tb, cfg, ce_dir, log=log)
            log(f"  cross-encoder rescored {len(idx):,} pairs ({len(idx) / max(len(C), 1):.3f})")
        df = pd.DataFrame({"s1_id": P1.id.values[C.s.values], "q_id": PQ.id.values[C.q.values],
                           "logit": C.logit.values, "ce": ce})
        df.to_pickle(cache)
        out.append(df)
        log(f"{country}: S1={len(S):,}  queries={len(Qc):,}  blocked={n_raw:,}  "
            f"candidates={len(C):,} ({len(C) / len(S):.2f} per S1)")

    D = (pd.concat(out, ignore_index=True) if out
         else pd.DataFrame(columns=["s1_id", "q_id", "logit", "ce"]))
    if "ce" not in D:
        D["ce"] = 0.0
    s_codes, s_uni = pd.factorize(D.s1_id)
    q_codes, q_uni = pd.factorize(D.q_id)
    final = D.logit.values + x.get("w", 0.0) * D.ce.values.astype(np.float64)
    ps, pq = select_matches(
        s_codes, q_codes, final,
        T=x["T"], b=x["b"], c=x["c"],
        tau_min=x.get("tau_min", 0.40),
        p_empty_mult=x.get("p_empty_mult", 1.05)
    )
    M = pd.DataFrame({"s1_id": np.asarray(s_uni)[ps], "q_id": np.asarray(q_uni)[pq]})

    ids = s1.entity_id.tolist()
    write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", ids,
                   D[["s1_id", "q_id"]])
    write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), "matched_entity_ids", ids, M)
    log(f"wrote outputs: {len(D):,} candidate pairs, {len(M):,} matches, "
        f"{(~s1.entity_id.isin(set(M.s1_id))).mean():.3f} of S1 predicted singleton")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "all"])
    for k, v in Config().dict().items():
        typ = (lambda t: str(t).lower() in ("1", "true", "yes", "y")) if isinstance(v, bool) else type(v)
        ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=v)
    a = vars(ap.parse_args())
    mode = a.pop("mode")
    cfg = Config(**a)
    if mode in ("train", "all"):
        train(cfg)
    if mode in ("predict", "all"):
        predict(cfg)
    log("done")


if __name__ == "__main__":
    main()
