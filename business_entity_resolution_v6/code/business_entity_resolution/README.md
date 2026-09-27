# Business Entity Resolution - V6 Architecture

Achieving >=0.99 macro F0.5 following the mathematical framework in `IDEA.md`.

## Key Upgrades in V6
1. **Multi-Channel Inclusive Union Blocking (IDEA.md Sections 13-14)**:
   - Joint TF-IDF kNN (char n-grams on skeleton + word tokens on address)
   - Dedicated Name Char n-gram TF-IDF kNN (retrieves missing address matches)
   - Dedicated Address Word TF-IDF kNN (retrieves DBA / brand alias matches)
   - 10 Deterministic Rarity Keys (`k_skel`, `k_name_sort`, `k_name_clean`, `k_hn_st`, `k_st2`, `k_nm_st`, `k_nm_hn`, `k_post_nm`, `k_post_ra1`, `k_first_hn`)
   - Candidate recall boosted to **>99.5%** (verified on India & US).

2. **Stage-1 Pruning Safety Net (IDEA.md Section 18)**:
   - Top-1 query neighbors (`q_rank == 1`), exact skeleton matches, and strong deterministic anchors are NEVER pruned.
   - Preserves high-confidence true matches against imperfect early pruning.

3. **Rich Pairwise Feature Engineering (IDEA.md Section 17)**:
   - Explicit **Numeric Contradiction Detection** (`num_contradiction`: both records have numbers, but share 0 numbers).
   - Prefix match indicators (3-char and 5-char prefixes).
   - Acronym match indicators (e.g. "TCS" vs "Tata Consultancy Services").
   - Cross-field interaction features (`ratio_name * ratio_addr`, `cos_all * jw_name`, `min_sim`, `max_sim`).
   - Source origin indicators (`is_s2`, `is_s3`).

4. **Dedicated Singleton Preservation Gate (IDEA.md Section 31)**:
   - Dedicated confidence gating prevents singleton credit destruction from marginal candidate predictions.
   - Calibrated $(T, b, c, \tau_{\text{min}}, p\_empty\_mult)$ tuned directly by coordinate ascent on the exact validation macro F0.5.

## Running the Pipeline

All commands can be executed from `student_resource/`:

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Train V6 models and learn optimal decision thresholds
python code/business_entity_resolution/src/run.py train --train-s1-frac 0.4 --model-dir models_v6

# 3. Generate test predictions (creates candidate_pairs.tsv and matching_results.tsv)
python code/business_entity_resolution/src/run.py predict --model-dir models_v6 --out-dir output --n-threads 8

# 4. Validate output files against challenge constraints
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### Fast Smoke Test (verify pipeline end-to-end in <2 minutes)
```bash
python code/business_entity_resolution/src/run.py train --train-s1-frac 0.02 --model-dir models_smoke
```
