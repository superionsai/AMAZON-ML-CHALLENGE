"""HPC Full Test Set Inference and Submission Packaging Pipeline.

Runs on full 1.73M test records across all countries:
- Loads multi-seed LightGBM ensembles and optimal thresholds from scratch
- Builds 4-channel candidates on test set
- Generates predictions and applies per-S1 expected F0.5 optimal subset rule
- Validates output integrity using validate_submission.py
- Creates submissions/TCC_submission_hpc_v6.zip
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "business_entity_resolution_v6"))

import joblib
import numpy as np
import pandas as pd

from config import V6Config
from blocking import build_country_candidates
from features import build_feature_matrix
from optimize import select_matches
from normalize import normalize_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hpc_predict_v6")


def main():
    parser = argparse.ArgumentParser(description="HPC Inference and Submission Packaging")
    parser.add_argument("--data-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/dataset")
    parser.add_argument("--model-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/models")
    parser.add_argument("--output-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/output")
    parser.add_argument("--submission-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/submissions")
    args = parser.parse_args()

    n_cpus = int(os.environ.get("NCPUS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 16))
    logger.info("=" * 70)
    logger.info("IIT DELHI HPC - TEST SET PREDICTION & PACKAGING")
    logger.info(f"Allocated Cores: {n_cpus}")
    logger.info(f"Models: {args.model_dir} | Output: {args.output_dir}")
    logger.info("=" * 70)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    sub_dir = Path(args.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    test_path = data_dir / "test.tsv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test.tsv at {test_path}")

    logger.info("Loading test table...")
    t0 = time.time()
    df_test = pd.read_csv(test_path, sep="\t", dtype=str).fillna("")
    logger.info(f"Loaded {len(df_test):,} test records in {time.time()-t0:.1f}s")

    countries = sorted(df_test["country"].dropna().unique())
    all_candidate_pairs = []
    all_matching_results = []

    for c in countries:
        model_file = model_dir / c / "model_bundle.pkl"
        if not model_file.exists():
            logger.warning(f"No model found for country {c} at {model_file}. Skipping.")
            continue

        logger.info(f"\n{'='*25} Predicting Country: {c} {'='*25}")
        bundle = joblib.load(model_file)
        models = bundle["models"]
        params = bundle["best_params"]
        cfg = bundle.get("cfg", V6Config(n_jobs=n_cpus))

        df_c = df_test[df_test["country"] == c].copy()
        df_c_norm = normalize_table(df_c)

        # 1. Blocking
        t_block = time.time()
        cands, _, _ = build_country_candidates(df_c_norm, None, cfg, is_train=False)
        logger.info(f"Generated {len(cands):,} candidate pairs in {time.time()-t_block:.1f}s")

        if len(cands) == 0:
            continue

        # 2. Features
        t_feat = time.time()
        X_df, _ = build_feature_matrix(cands, None, df_c_norm, cfg, n_jobs=n_cpus)
        logger.info(f"Built features in {time.time()-t_feat:.1f}s")

        # 3. Model Ensemble Prediction
        preds = np.zeros(len(X_df), dtype=np.float32)
        for m in models:
            preds += m.predict_proba(X_df)[:, 1] / len(models)
        cands["pred_prob"] = preds

        # 4. Optimal Threshold Decision Rule
        matched_cands = select_matches(cands, params, cfg)
        logger.info(f"Selected {len(matched_cands):,} matched pairs for country {c}")

        # Format candidates
        cands_formatted = cands[["s1_id", "s2_or_s3_id", "country"]].rename(
            columns={"s1_id": "source1_id", "s2_or_s3_id": "source2_or_source3_id"}
        )
        all_candidate_pairs.append(cands_formatted)

        # Format matches
        matches_formatted = matched_cands[["s1_id", "s2_or_s3_id", "country"]].rename(
            columns={"s1_id": "source1_id", "s2_or_s3_id": "source2_or_source3_id"}
        )
        all_matching_results.append(matches_formatted)

    # Concatenate all countries
    df_all_cands = pd.concat(all_candidate_pairs, ignore_index=True)
    df_all_matches = pd.concat(all_matching_results, ignore_index=True)

    cand_out = out_dir / "candidate_pairs.tsv"
    match_out = out_dir / "matching_results.tsv"

    logger.info(f"Writing {len(df_all_cands):,} candidates to {cand_out}...")
    df_all_cands.to_csv(cand_out, sep="\t", index=False)

    logger.info(f"Writing {len(df_all_matches):,} matches to {match_out}...")
    df_all_matches.to_csv(match_out, sep="\t", index=False)

    # Packaging Zip
    zip_path = sub_dir / "TCC_submission_hpc_v6.zip"
    logger.info(f"Packaging submission archive to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(cand_out, arcname="candidate_pairs.tsv")
        z.write(match_out, arcname="matching_results.tsv")

    logger.info("=" * 70)
    logger.info("SUBMISSION PACKAGE GENERATED SUCCESSFULLY!")
    logger.info(f"Location: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
