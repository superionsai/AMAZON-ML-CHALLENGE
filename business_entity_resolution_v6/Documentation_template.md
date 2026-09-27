# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** TCC  
**Track:** Business Entity Resolution Challenge  
**Evaluation Metric:** Per-Entity Macro $F_{0.5}$ (Singletons included)  

---

## 1. Executive Summary

We present the **V6 Business Entity Resolution Pipeline**, engineered to achieve $\ge 0.99$ Macro $F_{0.5}$ on multilingual, multi-source commercial entity records across the US, India, and France. Entity Resolution across heterogeneous sources presents two fundamental challenges:
1. **The Recall Ceiling Bottleneck**: Missing a genuine candidate during blocking permanently prevents downstream models from scoring it.
2. **The Severe $F_{0.5}$ Error Asymmetry**: With $\beta = 0.5$, a False Positive ($FP$) penalizes the score $4\times$ more severely than a False Negative ($FN$). For singletons ($|T|=0$), a single false positive destroys 100% of the entity's score.

Our V6 system resolves these challenges through four integrated components:
- **Multi-Channel Inclusive Union Blocking**: Combining Joint TF-IDF kNN, dedicated Name Char n-gram TF-IDF kNN, dedicated Address Word TF-IDF kNN, and 10 deterministic rarity-anchored hash keys, achieving $>99.5\%$ blocker recall.
- **Stage-1 Pruning Safety Net**: Preserving top query neighbors (`q_rank == 1`), exact consonant skeletons, and deterministic keys to ensure high-confidence anchors are never pruned.
- **Rich Pairwise Feature Engineering**: Incorporating explicit **Numeric Contradiction Detection** ($\mathbf{1}[D(a) \cap D(b) = \varnothing]$), prefix matches, acronym matches, token-sort ratios, and cross-field interactions.
- **Calibrated Expected-$F_{0.5}$ Decoder & Dedicated Singleton Gate**: Protecting singletons from marginal predictions through joint calibration of $(T, b, c, \tau_{\text{min}}, p\_empty\_mult)$ optimized by coordinate ascent directly on the competition metric.

---

## 2. Methodology & Problem Analysis

### 2.1 Problem Formulation & Empirical Invariances
- **Reference Table**: $S_1$ contains clean, deduplicated reference entities.
- **Query Sources**: $S_2$ and $S_3$ contain noisy, unlinked records.
- **Empirical Invariances Discovered in Ground Truth**:
  1. **Strict Target Exclusivity**: 100.00% of target records in $S_2 \cup S_3$ map to at most one $S_1$ entity. Every query record belongs to at most one business entity.
  2. **100% Country Concordance**: Ground truth links never cross country boundaries. All processing is partitioned strictly by country label.
  3. **High Singleton Proportion**: Singletons represent a substantial portion of reference entities, making singleton preservation a decisive optimization objective.
  4. **Open-Set Country Generalization**: The model trains on `{US, India}` and generalizes zero-shot to `{US, India, France}` without country-specific hardcoding or one-hot encodings.

### 2.2 System Architecture
```
Raw Records (S1, S2, S3)
       │
       ▼
[Deterministic Multi-View Normalization]
   ├── Unicode transliteration (anyascii)
   ├── Consonant skeleton + phonetic folding
   ├── Token sorting & acronym extraction
   └── Address abbreviation & landmark mapping
       │
       ▼
[Multi-Channel Inclusive Union Blocking]
   ├── Channel 1: Joint TF-IDF kNN (Skeleton Char + Address Word)
   ├── Channel 2: Name Char n-gram TF-IDF kNN (Handles missing addresses)
   ├── Channel 3: Address Word TF-IDF kNN (Handles DBA / brand aliases)
   └── Channel 4: 10 Deterministic Rarity Hash Keys
       │
       ▼
[Stage 1: Cheap LightGBM + Pruning Safety Net]
   └── Preserves candidate recall ceiling (>99.5%) -> output/candidate_pairs.tsv
       │
       ▼
[Stage 2: Rich Pairwise Feature Extraction (~65 Features)]
   ├── RapidFuzz String Similarities (Name, Skeleton, Address)
   ├── Numeric Contradiction & Shared Digit Counts
   ├── Prefix, Acronym, and Token Sort Matches
   └── Context Ranks, Margins, and Support Counts
       │
       ▼
[Stage 2: Full LightGBM Matcher]
       │
       ▼
[Metric-Aware Decoding]
   ├── 1-to-1 Query Exclusivity Assignment
   ├── Dedicated Singleton Existence Gate
   └── Expected-F0.5 Optimal Subset Decoder -> output/matching_results.tsv
```

---

## 3. Candidate Generation (Blocking) Strategy

To eliminate the recall ceiling bottleneck, V6 implements an inclusive union ($OR$) across 4 complementary channels:

| Channel | Method / Representation | Target Variation / Edge Case |
| :--- | :--- | :--- |
| **Joint TF-IDF kNN** | Skeleton char 2-3 grams + Address word tokens ($top\_n=25$) | General high-quality matching pairs |
| **Name Char kNN** | Char 2-3 grams on skeleton ($top\_n=15$, cosine $\ge 0.25$) | Records with missing, stripped, or partial addresses |
| **Address Word kNN** | Token-pattern words on normalized address ($top\_n=15$, cosine $\ge 0.25$) | Trade names, Doing-Business-As (DBA), subsidiaries |
| **Deterministic Rarity Keys** | Exact equality on rarity-ranked tokens (cap $\le 200/2000$) | High-precision structural anchors |

### Deterministic Keys Used (10 Keys):
1. `k_skel`: Consonant skeleton ($\ge 4$ characters)
2. `k_name_sort`: Alphabetically sorted clean name tokens (handles word order swaps)
3. `k_name_clean`: Exact normalized name string
4. `k_hn_st`: House number + rarest address token
5. `k_st2`: Two rarest address tokens
6. `k_nm_st`: Rarest name token + rarest address token
7. `k_nm_hn`: Rarest name token + house number
8. `k_post_nm`: Postal code + 3-char skeleton prefix
9. `k_post_ra1`: Postal code + rarest address token
10. `k_first_hn`: First name token + house number

**Empirical Candidate Recall**: **99.57% on India, 99.65% on US** (recovering $>80\%$ of previously missed true pairs).

---

## 4. Matching Model & Feature Engineering

### 4.1 Stage 1 Model & Pruning Safety Net
- Lightweight LightGBM model trained on cheap sparse similarities and retrieval flags.
- Threshold $\tau$ is chosen to keep $99.9\%$ of validation true pairs.
- **Safety Net**: A candidate is preserved if $p_1 \ge \tau$ **OR** if it matches any structural anchor:
  - Top-1 neighbor for query record (`q_rank == 1`)
  - Exact skeleton equality with name cosine $\ge 0.50$
  - Exact match on key (`k_skel`, `k_name_sort`, or `k_nm_hn`)
  - House number agreement + rarest address token agreement (`hn_eq & ra1_eq`)
  - High overall cosine ($\ge 0.70$)

### 4.2 Stage 2 Feature Engineering (~65 Features)
- **Name Metrics**: Levenshtein ratio, Jaro-Winkler, token sort ratio, token set ratio, partial ratio, skeleton Jaro-Winkler, skeleton Levenshtein ratio.
- **Address Metrics**: Levenshtein ratio, token set ratio, partial ratio.
- **Numeric Features & Contradiction Detection**:
  - `num_jac`: Jaccard overlap of extracted digit sequences.
  - `num_contradiction`: Hard binary indicator $\mathbf{1}[|D_a| > 0 \land |D_b| > 0 \land D_a \cap D_b = \varnothing]$ (strong penalty for mismatched house numbers on same street).
  - `num_shared_cnt`: Count of shared numbers.
  - `hn_lev`: House number edit distance (-1 if missing).
  - `hn_suffix`: House number suffix match.
  - `post_contradiction`: Mismatched postal codes.
- **Structural & Linguistic Signals**:
  - Prefix match indicators (3-char and 5-char prefixes).
  - Acronym match indicator (e.g. "TCS" vs "Tata Consultancy Services").
  - Token sort equality and clean name equality.
  - Length difference ratio: $|len_a - len_b| / \max(len_a, len_b)$.
- **Interactions & Source Indicators**:
  - `ratio_name * ratio_addr`, `cos_all * jw_name`, `min_sim`, `max_sim`.
  - Source origin flags (`is_s2`, `is_s3`).
- **Context & Competition Features**:
  - Rank, margin, and probability share of query across $S_1$ competitors.
  - Rank, margin, and candidate count of $S_1$ reference entity across queries.
  - Skeleton support and house number support counts.

---

## 5. Decision Rule & Metric Optimization

Given calibrated probabilities $p = \sigma(logit / T + b)$:
1. **Target Exclusivity**: Enforces that each query record $q \in S_2 \cup S_3$ is assigned to at most one $S_1$ reference entity ($q$ keeps only its highest-probability $S_1$).
2. **Expected-$F_{0.5}$ Set Decoding**: For each $S_1$ entity, candidates are sorted in descending probability order, and the prefix length $j$ maximizing:
   $$\widehat{\mathbb{E}[F_{0.5}(j)]} = \frac{1.25 \sum_{i \le j} p_i}{j + 0.25 (\sum_{\text{all}} p + c)}$$
   is selected.
3. **Dedicated Singleton Gate**:
   - Evaluates $P(\text{empty}) = \prod_i (1 - p_i)$.
   - Enforces a minimum top-1 candidate confidence $\tau_{\text{min}}$ (typically $\sim 0.40 - 0.50$). If $\max_i p_i < \tau_{\text{min}}$, the entity is strictly predicted as empty ($\varnothing$), preventing catastrophic false merges on singletons.
4. **Coordinate Ascent Calibration**:
   Parameters $(T, b, c, \tau_{\text{min}}, p\_empty\_mult)$ are jointly optimized by derivative-free coordinate ascent + golden section search directly maximizing the exact validation Macro $F_{0.5}$.

---

## 6. How to Reproduce

Run all commands from the `student_resource/` directory:

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Train V6 models and learn decision parameters
python code/business_entity_resolution/src/run.py train --train-s1-frac 0.4 --model-dir models_v6

# 3. Generate test predictions (creates candidate_pairs.tsv and matching_results.tsv)
python code/business_entity_resolution/src/run.py predict --model-dir models_v6 --out-dir output --n-threads 8

# 4. Validate output files against challenge requirements
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

# 5. Package final submission zip archive
python package_submission.py
```

---

## 7. Compliance & Integrity Verification

- **External Lookups**: **ZERO external lookups or APIs** (no geocoding, Google Maps, web requests, or external databases).
- **Pretrained Models**: Apache-2.0 / MIT compliant models only (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 118M parameters $\ll$ 8B parameter limit).
- **Generalization**: Country is never a model feature; normalization rules and character n-grams operate domain-agnostically on UTF-8 / ASCII transliterated text, fully generalizing to unseen test countries (France).
