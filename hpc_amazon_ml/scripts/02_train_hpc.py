"""HPC Full-Scale Training Pipeline for Business Entity Resolution V6.

Designed specifically for IIT Delhi HPC (PADUM) cluster:
- High RAM (64 GB - 128 GB)
- High multi-core parallelism (24 - 48 cores via os.sched_getaffinity / $NCPUS)
- 100% dataset training (train_s1_frac = 1.0)
- Deep LightGBM ensemble with multi-seed bagging
- Coordinate-ascent decision threshold optimization on exact macro F0.5
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "business_entity_resolution_v6"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from anyascii import anyascii

# Import V6 components
from config import V6Config
from blocking import build_country_candidates
from features import build_feature_matrix
from optimize import optimize_thresholds_for_country, select_matches
from normalize import normalize_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hpc_train_v6")


def get_hpc_core_count() -> int:
    """Accurately determine CPU core allocation inside PBS Pro job.
    Avoids Gotcha #12 where nproc / cpu_count() reports 1 inside PBS."""
    if "NCPUS" in os.environ:
        try:
            return int(os.environ["NCPUS"])
        except ValueError:
            pass
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    return max(1, os.cpu_count() or 4)


def main():
    parser = argparse.ArgumentParser(description="HPC Business Entity Resolution V6 Training")
    parser.add_argument("--data-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/dataset",
                        help="Path to directory containing train.tsv, test.tsv, train_labels.tsv")
    parser.add_argument("--output-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/models",
                        help="Path to save trained models and thresholds")
    parser.add_argument("--train-s1-frac", type=float, default=1.0,
                        help="Fraction of S1 training entities to use (1.0 = 100% full dataset)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Validation split fraction")
    parser.add_argument("--knn-k", type=int, default=25,
                        help="kNN candidates per channel")
    parser.add_argument("--num-leaves", type=int, default=127,
                        help="LightGBM complexity: number of leaves")
    parser.add_argument("--n-estimators", type=int, default=1500,
                        help="Number of boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.03,
                        help="Boosting learning rate")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of bagged model seeds for ensemble")
    args = parser.parse_args()

    n_cpus = get_hpc_core_count()
    logger.info("=" * 70)
    logger.info("IIT DELHI HPC (PADUM) - BUSINESS ENTITY RESOLUTION V6")
    logger.info(f"Allocated CPU Cores: {n_cpus}")
    logger.info(f"Dataset Directory: {args.data_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Training S1 Fraction: {args.train_s1_frac * 100:.1f}%")
    logger.info(f"LightGBM Leaves: {args.num_leaves} | Estimators: {args.n_estimators} | Seeds: {args.n_seeds}")
    logger.info("=" * 70)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.tsv"
    labels_path = data_dir / "train_labels.tsv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.tsv at {train_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing train_labels.tsv at {labels_path}")

    logger.info("Loading training tables...")
    t0 = time.time()
    df_train = pd.read_csv(train_path, sep="\t", dtype=str).fillna("")
    df_labels = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
    logger.info(f"Loaded {len(df_train):,} train records and {len(df_labels):,} labels in {time.time()-t0:.1f}s")

    # Group by country
    countries = sorted(df_train["country"].dropna().unique())
    logger.info(f"Countries to process: {countries}")

    summary_metrics = {}

    for c in countries:
        logger.info(f"\n{'='*30} Processing Country: {c} {'='*30}")
        df_c = df_train[df_train["country"] == c].copy()
        df_labels_c = df_labels[df_labels["country"] == c].copy()
        logger.info(f"Country {c}: {len(df_c):,} records, {len(df_labels_c):,} label pairs")

        # 1. Normalize
        t_norm = time.time()
        df_c_norm = normalize_table(df_c)
        logger.info(f"Normalized {len(df_c_norm):,} records in {time.time()-t_norm:.1f}s")

        # 2. Candidate Generation (Blocking)
        cfg = V6Config(
            knn_k=args.knn_k,
            train_s1_frac=args.train_s1_frac,
            val_frac=args.val_frac,
            n_jobs=n_cpus
        )
        t_block = time.time()
        candidates, true_pairs, split_info = build_country_candidates(
            df_c_norm, df_labels_c, cfg, is_train=True
        )
        logger.info(f"Generated {len(candidates):,} candidate pairs in {time.time()-t_block:.1f}s")

        # 3. Feature Extraction
        t_feat = time.time()
        X_df, y = build_feature_matrix(candidates, true_pairs, df_c_norm, cfg, n_jobs=n_cpus)
        logger.info(f"Extracted {X_df.shape[1]} features for {len(X_df):,} candidates in {time.time()-t_feat:.1f}s")

        # 4. Train / Val Split
        train_mask = candidates["is_train_split"].values
        val_mask = ~train_mask
        X_train, y_train = X_df[train_mask], y[train_mask]
        X_val, y_val = X_df[val_mask], y[val_mask]
        logger.info(f"Train set: {len(X_train):,} pairs (pos: {int(y_train.sum()):,}) | Val set: {len(X_val):,} pairs (pos: {int(y_val.sum()):,})")

        # 5. Multi-Seed Bagged LightGBM Training
        models = []
        val_preds_list = []
        base_seeds = [42, 100, 2026][:args.n_seeds]

        for s_idx, seed in enumerate(base_seeds):
            logger.info(f"Training LightGBM model {s_idx+1}/{len(base_seeds)} (seed={seed})...")
            model = lgb.LGBMClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                max_depth=12,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.80,
                min_child_samples=50,
                random_state=seed,
                n_jobs=n_cpus,
                importance_type="gain"
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=200)]
            )
            models.append(model)
            val_p = model.predict_proba(X_val)[:, 1]
            val_preds_list.append(val_p)

        # Ensemble validation predictions
        val_preds = np.mean(val_preds_list, axis=0)
        candidates_val = candidates[val_mask].copy()
        candidates_val["pred_prob"] = val_preds

        # 6. Coordinate Ascent Threshold Optimization
        t_opt = time.time()
        best_params, best_f05 = optimize_thresholds_for_country(
            candidates_val, true_pairs, cfg
        )
        logger.info(f"Country {c} Best Val Macro F0.5 = {best_f05:.5f} (optimized in {time.time()-t_opt:.1f}s)")
        logger.info(f"Optimal Thresholds: {best_params}")

        # Save model ensemble and parameters
        country_save_dir = out_dir / c
        country_save_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({
            "models": models,
            "best_params": best_params,
            "best_f05": best_f05,
            "feature_names": list(X_df.columns),
            "cfg": cfg
        }, country_save_dir / "model_bundle.pkl")

        summary_metrics[c] = {
            "val_f05": best_f05,
            "pairs": len(candidates),
            "features": X_df.shape[1],
            "params": best_params
        }

    logger.info("\n" + "=" * 70)
    logger.info("FULL DATASET HPC TRAINING SUMMARY")
    total_f05 = np.mean([m["val_f05"] for m in summary_metrics.values()])
    for c, m in summary_metrics.items():
        logger.info(f"Country {c:15s} | Val Macro F0.5: {m['val_f05']:.5f} | Pairs: {m['pairs']:,}")
    logger.info(f"Mean Macro F0.5 across all countries: {total_f05:.5f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
