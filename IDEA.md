You are modifying the existing `business_entity_resolution_v5` solution.

Current leaderboard score:

```text
0.892 macro F_0.5
```

Target:

```text
maximize leaderboard macro F_0.5; aim toward 0.99
```

Do NOT rewrite the project from scratch.

Preserve working V5 components unless an ablation proves replacement is better.

Required architecture:

```text
multi-view normalization
→ multi-channel high-recall retrieval
→ structured stage-1/stage-2 matcher
→ risk-based cross-encoder reranking
→ OOF-trained fusion model
→ target-owner/group reasoning
→ source-aware F0.5 decoder
```

The current V5 has the following known limitations that MUST be addressed:

```text
1. stage-1 pruning intentionally targets only ~0.998 positive recall
2. validation/training target universe is incomplete when S1 is sampled
3. S2/S3 source identity is mostly discarded
4. name/address normalization is over-destructive
5. DBA handling destroys one of the name identities
6. retrieval mixes name/address with fixed 0.6/0.4 attenuation
7. dense embedding is feature-only, not retrieval
8. CE is routed only through |LightGBM logit| < 3
9. CE training is uncertainty-biased rather than hard-error-biased
10. CE input lacks explicit field/source structure
11. CE and LightGBM are fused with a scalar weighted sum
12. model is binary-pair trained but inference behaves like target ownership/ranking
13. no source-conditioned evidence attenuation
14. candidate oracle must be evaluated under a full-density realistic validation universe
```

Implement the following.

# 1. FREEZE V5 BASELINE

Before changes:

```text
- preserve current V5 config
- preserve current model artifacts
- preserve current validation predictions if available
- save baseline metrics
```

Create:

```text
artifacts/v5_baseline_metrics.json
```

Include:

```text
macro_f05
candidate_pair_recall
candidate_full_entity_recall
candidate_any_entity_recall
oracle_macro_f05
singleton_f05
non_singleton_f05
S2 metrics
S3 metrics
mean candidates
p50/p90/p99 candidates
zero-candidate fraction
FP count
FN count
blocker FN count
matcher FN count
```

Do not compare later versions against leaderboard score alone.

# 2. VALIDATION SPLIT

Validation MUST be grouped by Source-1 entity.

Never split pairs randomly.

Use deterministic folds:

```python
RANDOM_SEED = 42
N_FOLDS = 5
```

If full 5-fold CV is too expensive:

```text
1 fixed 80/20 grouped split for large models
5-fold OOF for cheap structured models
```

Stratify approximately by:

```text
country
singleton/non-singleton
true-match cardinality bucket: 0,1,2,3+
```

No S1 entity may occur in both train and validation.

# 3. CRITICAL VALIDATION-DENSITY FIX

If validation uses only a sampled subset of S1 records, the retrieval universe MUST still contain ALL relevant S2 and S3 records.

Do NOT construct validation target universe as:

```text
positive targets of sampled S1
+
unmatched distractors
```

This removes hard negatives belonging to other S1 entities.

Correct validation setup:

```text
validation S1 = held-out S1 subset
candidate target universe = full train_source2 + full train_source3
```

Every S2/S3 record belonging to non-validation S1 entities remains available as a competitor.

This is mandatory.

# 4. GROUND-TRUTH INDEXES

Build:

```python
truth_by_s1: dict[str, set[str]]
owner_by_target: dict[str, str]
```

Before using target exclusivity, verify:

```python
max(len(owners[target])) == 1
```

for every GT S2/S3 target.

Store:

```text
target_exclusivity_verified = true/false
```

Do not hard-enforce one-owner logic unless training GT verifies it.

# 5. ORACLE METRICS

For every validation S1:

```python
C_i = final candidates entering matcher
T_i = true target set
```

Compute:

```text
pair_recall
any_entity_recall
full_entity_recall
```

Definitions:

```python
pair_recall =
    total_true_links_present / total_true_links

any_entity_recall =
    mean(bool(C_i & T_i)) over T_i != empty

full_entity_recall =
    mean(T_i <= C_i) over T_i != empty
```

Compute candidate-oracle prediction:

```python
oracle_pred_i = C_i & T_i
```

Then exact macro F0.5.

This is the hard ceiling of the matcher.

Before spending time on CE improvements:

```text
if oracle macro F0.5 < ~0.995:
    fix retrieval/pruning first
```

Target:

```text
pair recall >= 0.9995 if feasible
full-entity recall as close to 1.0 as feasible
oracle macro F0.5 >= 0.995
```

Do NOT intentionally target only 0.998 pair recall if the objective is ~0.99 final F0.5.

# 6. MULTI-VIEW NORMALIZATION

Replace one destructive normalization with field-specific views.

Implement:

```python
normalize_name_light()
normalize_name_legal()
normalize_name_phonetic()
normalize_address_generic()
normalize_address_country_aux()
extract_dba_names()
```

Never modify raw values.

For each business create:

```text
name_raw
name_light
name_legal
name_phonetic

address_raw
address_generic
address_country_aux

legal_name
trade_name
```

# 7. NAME NORMALIZATION

`name_light`:

```text
Unicode NFKC
casefold
normalize punctuation
& ↔ and auxiliary normalization
collapse whitespace
retain Unicode letters
retain digits
```

Do NOT remove:

```text
services
service
solutions
enterprises
enterprise
associates
group
international
center
centre
```

from the main name representation.

`name_legal` may remove only actual organization-form tokens supported by data, e.g.:

```text
pvt
private
ltd
limited
llc
llp
inc
incorporated
corp
corporation
company
co
plc
```

Keep `name_light` in parallel.

# 8. DBA HANDLING

Do NOT replace:

```text
ABC Holdings DBA Sunshine Cafe
```

with only:

```text
Sunshine Cafe
```

Parse:

```python
legal_name = "ABC Holdings"
trade_name = "Sunshine Cafe"
```

Recognize markers such as:

```text
dba
d/b/a
doing business as
trading as
t/a
```

where observed in data.

Pair features MUST include:

```text
legal↔legal similarity
legal↔trade similarity
trade↔legal similarity
trade↔trade similarity
max cross-name similarity
```

If no DBA exists:

```text
trade_name = ""
```

# 9. ADDRESS NORMALIZATION

Do NOT apply legal-name token stripping to address.

`address_generic` must work for:

```text
US
India
France
unseen countries
```

Use:

```text
Unicode normalization
punctuation normalization
whitespace normalization
digit preservation
safe universal abbreviations only
```

Country-specific US/India dictionaries may exist only as:

```text
address_country_aux
```

Never replace the generic view.

Do not make ambiguous global replacements such as:

```text
saint → st
street → st
```

inside the same destructive representation.

# 10. RETRIEVAL: REMOVE FIXED 0.6/0.4 DEPENDENCE

Do NOT make the sole retrieval representation:

```python
0.6 * name + 0.4 * address
```

because it attenuates single-field strong matches.

Generate candidates independently.

For each source separately:

```text
C_name_char
C_name_word
C_address_char
C_address_word
C_combined
C_exact
C_numeric
C_rare
```

Then:

```python
C = union(all channels)
```

# 11. NAME CHAR RETRIEVAL

Use true IDF weighting.

Preferred:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    sublinear_tf=True,
    norm="l2",
)
```

If inverted-index implementation is required for scale:

```python
idf(token) = log((N + 1) / (df(token) + 1))
```

Candidate score:

```python
score[q,d] += idf(token)
```

Do NOT use fixed:

```python
+1
+3
```

weights.

# 12. NAME WORD RETRIEVAL

Use:

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
)
```

or equivalent IDF-weighted inverted index.

Rare tokens must contribute more than generic tokens.

# 13. ADDRESS RETRIEVAL

Independent retrieval using:

```text
address_generic char TF-IDF
address_generic word TF-IDF
```

This MUST generate candidates even when name retrieval fails.

# 14. NUMERIC / RARE TOKEN RETRIEVAL

Extract:

```text
numeric tokens
first numeric token
last numeric token
alphanumeric building tokens
long/postal-like numeric tokens
rare name tokens
rare address tokens
```

Use IDF/frequency.

Retrieve candidates sharing distinctive evidence.

# 15. EXACT RETRIEVAL

Indexes:

```text
exact name_light
exact name_legal
exact legal_name
exact trade_name
exact address_generic
exact rare-name + numeric-signature
```

If a key is highly frequent, rank internally using other evidence instead of returning unlimited candidates.

# 16. OPTIONAL DENSE RETRIEVAL

Current V5 embedding model is feature-only.

First measure oracle without dense retrieval.

If lexical/multi-field retrieval fails target oracle requirement, add dense retrieval:

```text
C_dense
```

using the existing compliant multilingual encoder or another verified MIT/Apache-2.0 model.

Do NOT add dense retrieval merely for complexity.

Measure:

```text
oracle before dense
oracle after dense
candidate-count change
```

Keep only if recall improves enough to justify cost.

# 17. RETRIEVAL K SEARCH

Do not hard-code a single K without validation.

Test:

```text
5
10
20
30
50
75
```

per major channel.

Measure incremental union recall:

```text
name only
+ address
+ exact
+ numeric/rare
+ dense
```

Select smallest configuration satisfying target oracle.

# 18. STAGE-1 PRUNING SAFETY NET

Do NOT prune using only:

```python
p_stage1 >= threshold
```

Use:

```python
keep =
    (p_stage1 >= tau_prune)
    OR (rank_for_target <= K_TARGET_SAFE)
    OR (rank_for_s1 <= K_S1_SAFE)
    OR deterministic_high_recall_key
```

Initial search space:

```text
K_TARGET_SAFE ∈ {2,3,5}
K_S1_SAFE ∈ {2,3,5}
```

Tune using candidate oracle.

Stage-1 prune objective:

```text
minimum compute subject to near-perfect downstream candidate recall
```

not classifier precision.

# 19. SOURCE IDENTITY

Preserve candidate source:

```text
S2
S3
```

as an explicit field throughout.

Structured features:

```text
candidate_is_S2
candidate_is_S3
```

Neural input includes:

```text
[SOURCE_A=S1]
[SOURCE_B=S2]
```

or:

```text
[SOURCE_B=S3]
```

Evaluate:

```text
one shared model + source feature
```

against:

```text
source-specific models
```

At minimum tune separate final thresholds/calibration for S2 and S3.

# 20. STRUCTURED PAIR FEATURES

Expand V5 features.

NAME:

```text
char_tfidf_light
char_tfidf_legal
word_tfidf_light
word_tfidf_legal

rapidfuzz_ratio
rapidfuzz_WRatio
token_sort
token_set

jaro_winkler
token_jaccard
token_containment

exact_name_light
exact_name_legal

legal_legal_sim
legal_trade_sim
trade_legal_sim
trade_trade_sim
max_dba_sim

shared_rare_name_idf
name_length_ratio
name_token_count_diff
name_digit_jaccard
name_digit_conflict
```

ADDRESS:

```text
char_tfidf_generic
word_tfidf_generic
char_tfidf_country_aux

rapidfuzz_address_ratio
address_token_set
address_jaccard
address_containment

shared_rare_address_idf
address_length_ratio
address_token_count_diff
```

NUMERIC:

```text
numeric_jaccard
numeric_overlap_count
numeric_set_equal

first_number_equal
first_number_conflict
last_number_equal
last_number_conflict

numeric_sequence_similarity

postal_like_equal
postal_like_conflict

alphanumeric_number_equal
alphanumeric_number_conflict
```

OTHER:

```text
country_equal
candidate_source
name_missing
address_missing

name_x_address
max_name_address
min_name_address

retrieval_channel_count
retrieved_by_* flags
retrieval ranks
retrieval scores
```

# 21. RELIABILITY / ATTENUATION FEATURES

Do NOT use fixed global:

```text
name weight = 0.6
address weight = 0.4
```

Compute reliability:

```text
name rarity / IDF
name token count
name genericness
address rarity / IDF
address token count
numeric token count
missingness
```

Let downstream model learn effective attenuation.

Also implement empirical source-conditioned likelihood evidence.

For each source `s`, feature/bin `b`:

```python
m = P(feature_bin=b | match, source=s)
u = P(feature_bin=b | nonmatch, source=s)

llr = log((m + eps) / (u + eps))
```

Compute from TRAIN folds only.

Never derive LLR statistics using validation labels.

Useful binned evidence:

```text
name similarity
address similarity
numeric agreement
numeric contradiction
rare-token overlap
country equality
```

Add:

```text
source_name_llr
source_address_llr
source_numeric_llr
combined_llr
```

to fusion model.

# 22. TRAINING DATA

All GT positives must be available.

Do not use arbitrary target prefixes that drop positives.

Construct:

```text
all_positive_pairs
```

from full GT.

For hard negatives use actual retrieval/matcher competitors.

Negative categories:

```text
top wrong target-owner candidate
top wrong S1-side candidate
high name + wrong address
high address + wrong name
same exact/legal-stripped name but wrong entity
numeric-conflict near match
same rare token but wrong entity
high Stage-1 false positive
high Stage-2 false positive
cross-encoder high-score false positive from previous mining round
```

Avoid wasting most training examples on trivial random negatives.

# 23. STAGE-1 / STAGE-2 LIGHTGBM

Keep structured models.

Use deterministic parameters and early stopping.

Do not select model using pair accuracy.

Model selection uses validation macro F0.5 after decoding.

Save OOF logits, not only probabilities.

# 24. CROSS-ENCODER INPUT

Replace:

```text
name ; address
```

with explicit structured text:

```text
[SOURCE_A] S1
[NAME_A] ...
[LEGAL_NAME_A] ...
[TRADE_NAME_A] ...
[ADDRESS_A] ...
[COUNTRY_A] ...

[SOURCE_B] S2_OR_S3
[NAME_B] ...
[LEGAL_NAME_B] ...
[TRADE_NAME_B] ...
[ADDRESS_B] ...
[COUNTRY_B] ...
```

Keep input short.

Recommended initial:

```text
max_length = 128
```

Increase only if truncation analysis shows meaningful field loss.

# 25. CROSS-ENCODER TRAINING SET

Current uncertainty-only sampling is insufficient.

CE training must include:

```text
ALL feasible positives
```

plus targeted hard negatives.

Hard-negative priority:

```text
1. high-confidence structured-model FP
2. wrong candidate ranked immediately below true owner
3. exact/similar name but different entity
4. strong address but wrong entity
5. primary-number contradiction with high text similarity
6. reciprocal/ownership ambiguous candidates
7. ordinary blocker hard negatives
```

Use trivial random negatives only as a small minority.

# 26. CROSS-ENCODER HARD-NEGATIVE MINING

Implement at least one iterative mining round.

Round 1:

```text
train CE_1
```

Score training/OOF candidate pairs.

Mine:

```python
y == 0
AND CE_score among highest negatives
```

plus:

```python
y == 0
AND final_fusion_score high
```

Add/reweight them.

Train:

```text
CE_2
```

Compare OOF macro F0.5.

Keep CE_2 only if improved.

# 27. CROSS-ENCODER LOSS

Do not use only binary BCE if target exclusivity is verified.

Use:

```text
L = L_BCE + λ_rank * L_rank
```

For target query `q`:

```text
positive owner = s+
hard negative owner = s-
```

Margin loss:

```python
L_rank = max(0, margin - z_pos + z_neg)
```

Initial search:

```text
margin ∈ {0.2, 0.5, 1.0}
lambda_rank ∈ {0.1, 0.25, 0.5, 1.0}
```

Evaluate OOF.

Alternative acceptable:

```text
pairwise logistic ranking loss
```

Do not add ranking loss if target exclusivity is false.

# 28. CROSS-ENCODER ROUTING

Remove routing based only on:

```python
abs(LGBM_logit) < 3
```

This misses high-confidence wrong predictions.

Build risk-based route set as union of:

```text
top K candidate owners per target
top K targets per S1
low Stage-2 margin
high name/address disagreement
primary-number conflict with high name similarity
DBA cases
non-ASCII/transliteration cases
high target-ownership ambiguity
structured-model uncertainty
structured-model/embedding disagreement
high-value exact-name collisions
```

Initial:

```text
K ∈ {2,3}
```

Route all validation positives during CE training/evaluation so CE recall is measurable.

For test inference, use tuned risk routing.

Report:

```text
% candidate pairs sent to CE
% S1 entities using CE
CE recall over structured-model errors
```

# 29. DO NOT ACCEPT BY CE DIRECTLY

CE produces:

```text
ce_logit
ce_probability
```

These become features.

Do not simply:

```python
if CE_prob > threshold:
    match
```

# 30. LEARNED FUSION MODEL

Delete final decision of form:

```python
final_score = lgbm_logit + w * ce_logit
```

Replace with an OOF-trained fusion model.

Inputs:

```text
stage1_logit
stage2_logit

ce_logit
ce_probability

embedding_cosine

abs(stage2_logit - ce_logit)
sign_disagreement
model_agreement_count

all important structured features

source
country_equal

retrieval ranks
retrieval channel count

source-conditioned LLR features
```

Train only using OOF base-model predictions for training rows.

Never train fusion model using in-sample predictions from base models.

Preferred:

```text
LightGBM fusion classifier
```

Also test logistic fusion baseline.

Select by OOF macro F0.5 after final decoder.

# 31. FULL CE COVERAGE FEATURE HANDLING

Not every candidate will be CE-routed.

Fusion features must include:

```text
ce_was_run
```

For non-routed pairs:

```text
ce_logit = missing
```

Use native LightGBM missing handling.

Do not fill non-routed CE score with zero unless zero explicitly means neutral and validation proves it.

# 32. TARGET-OWNER FORMULATION

If training verifies target exclusivity, model target ownership explicitly.

For every target `q`:

```text
candidate S1 owners = s1...sk
```

Features:

```text
final pair score
pair rank for target
top1 score
top2 score
top1-top2 margin
current-vs-best-other margin
CE score
retrieval evidence
```

Train/evaluate either:

```text
LightGBMRanker grouped by target
```

or a target-owner classifier using relative features.

Required relative features:

```python
target_rank
target_best_score
target_second_score
target_margin_to_best_other
is_target_top1
```

# 33. RECIPROCAL FEATURES

For pair `(s,q)` compute:

```text
rank(q among s candidates)
rank(s among q owner candidates)

is_s1_top1_for_target
is_target_top1_for_s1
is_reciprocal_top1
is_reciprocal_top3
```

Add to fusion/ownership model.

# 34. NONE OPTION / TARGET OWNERSHIP

If exclusivity is verified, every target conceptually has:

```text
one S1 owner
OR
no matching S1
```

Do not force assignment.

Tune an ownership rejection threshold.

Conceptually:

```python
if best_owner_score < tau_owner_none:
    owner = NONE
else:
    owner = best_candidate
```

Tune `tau_owner_none` using OOF macro F0.5.

Test source-specific:

```text
tau_owner_none_S2
tau_owner_none_S3
```

# 35. S1-SIDE GROUP FEATURES

For every candidate pair add:

```text
s1_pair_rank
s1_best_score
s1_second_score
s1_top1_top2_margin
number_above_0.5
number_above_0.8
number_above_0.95
```

Threshold counts should use OOF calibrated probabilities where possible.

# 36. SINGLETON MODEL

Train only using OOF pair/fusion predictions.

Target:

```python
has_match = int(len(T_i) > 0)
```

Features:

```text
top1 final probability
top2 probability
top1-top2 margin

best S2 probability
best S3 probability

best name score
best address score
best numeric evidence

candidate count
CE-routed count
reciprocal-top1 evidence

number of high-confidence candidates
```

Compare against simple final-threshold decoder.

Keep singleton model only if macro F0.5 improves.

# 37. CARDINALITY MODEL

Optional after singleton model.

Predict:

```text
0
1
2
3+
```

using group features.

Use predicted cardinality as decoder feature, not initially as a hard constraint.

Keep only if OOF macro F0.5 improves.

# 38. F0.5 DECODER

Implement exact metric:

```python
def entity_f05(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0

    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)

    if tp == 0:
        return 0.0

    return 1.25 * precision * recall / (
        0.25 * precision + recall
    )
```

Final metric:

```python
macro_f05 = mean(entity_f05(...))
```

Do not micro-average.

# 39. THRESHOLD SEARCH

Cache OOF predictions.

Tune without retraining.

Search:

```text
pair acceptance threshold
owner NONE threshold
singleton threshold
S2 threshold
S3 threshold
```

Use coarse then fine search.

Example:

```text
coarse: step 0.01
fine around optimum: step 0.001
```

Do not assume 0.5.

# 40. SOURCE-SPECIFIC DECODING

At minimum test:

```text
shared threshold
```

vs:

```text
tau_S2
tau_S3
```

Also test source-conditioned ownership thresholds.

Use whichever improves OOF macro F0.5.

# 41. CALIBRATION

Using OOF predictions only, compare:

```text
raw fusion probability
Platt scaling
isotonic calibration
```

Evaluate:

```text
macro F0.5 after decoder
Brier score
reliability buckets
```

Calibration is required only if it improves decoding.

# 42. ERROR TAXONOMY

For every validation error classify:

```text
BLOCKER_MISS
STAGE1_PRUNE_MISS
STRUCTURED_FALSE_NEGATIVE
CE_FALSE_NEGATIVE
FUSION_FALSE_NEGATIVE
DECODER_FALSE_NEGATIVE

STRUCTURED_FALSE_POSITIVE
CE_FALSE_POSITIVE
FUSION_FALSE_POSITIVE
OWNER_FALSE_POSITIVE
SINGLETON_FALSE_POSITIVE
```

Save counts.

Also save examples for:

```text
DBA
transliteration/non-ASCII
same-name different-address
different-name same-address
primary-number conflict
generic business names
source S2
source S3
country US
country India
```

# 43. HARD ERROR FILES

Write:

```text
artifacts/errors/high_confidence_false_positives.tsv
artifacts/errors/high_confidence_false_negatives.tsv
artifacts/errors/blocker_misses.tsv
artifacts/errors/owner_conflicts.tsv
```

Include:

```text
S1 ID
target ID
source
names
addresses
truth
stage1 score
stage2 score
CE score
fusion score
ranks
numeric conflict features
retrieval channels
```

# 44. FRANCE ROBUSTNESS

Do not train country-specific one-hot logic that cannot represent France.

Primary features:

```text
country_equal
generic Unicode text
generic address features
numeric features
multilingual CE
```

Country-specific normalization only auxiliary.

Run diagnostics:

```text
train US → validate India
train India → validate US
```

Compare degradation.

If leave-country-out collapses, reduce dependence on country-specific preprocessing.

# 45. MODEL SIZE / GPU CONSTRAINT

Local hardware:

```text
6 GB NVIDIA GPU
```

For current MiniLM CE:

```text
FP16
max_length 128
small batch
gradient accumulation
```

Do not create OOM-prone training.

Prefer:

```text
batch_size 4–16
```

depending on actual memory.

Enable:

```text
gradient_checkpointing
```

only if needed.

# 46. OPTIONAL STRONGER RERANKER

Do NOT replace MiniLM immediately.

First implement:

```text
better training data
risk routing
structured CE input
ranking loss
learned fusion
```

Then test a stronger multilingual Apache/MIT-compatible reranker ONLY on validation hard cases.

Candidate model:

```text
BAAI/bge-reranker-v2-m3
```

Verify actual license before inclusion.

Because it is substantially larger:

```text
use for inference/reranking hard tail first
do not full-finetune on 6GB GPU unless memory-safe method is verified
```

Compare on identical validation routed pairs.

Metrics:

```text
CE pair PR-AUC
hard-negative accuracy
final macro F0.5 after fusion
runtime
```

Keep stronger reranker only if final macro F0.5 improves sufficiently.

# 47. OPTIONAL MULTI-SOURCE TRIANGLE SUPPORT

After pair/fusion/owner system works, test local S2↔S3 support.

For S1 entity `s` and S3 candidate `y`:

```python
triangle_support_s3 =
    max over strong S2 candidates x:
        score(s,x) * score(x,y)
```

Analogous for S2.

Do NOT do global transitive closure.

Features only:

```text
max_cross_source_support
number_strong_cross_source_supporters
triangle_reciprocal
```

Only calculate inside local candidate neighborhood.

Keep only if OOF macro F0.5 improves.

# 48. NO EXTERNAL ENTITY DATA

Strictly forbidden:

```text
Google
Google Maps
geocoding
business registry
company databases
commercial ER APIs
internet business lookup
```

Allowed evidence:

```text
provided challenge data
model parameters satisfying challenge license requirements
statistics learned from provided training data
```

# 49. EXPERIMENT ORDER

Do not implement every advanced component simultaneously.

Run in this exact order:

```text
V6-A
full-density validation fix
+ correct oracle metrics

V6-B
multi-view normalization
+ DBA split
+ source preservation

V6-C
independent name/address/exact/numeric retrieval
+ stage1 rank safety-net
+ optimize candidate oracle

V6-D
full-positive / hard-negative structured training
+ source features
+ expanded numeric features

V6-E
rebuild CE dataset
+ structured CE input
+ risk routing

V6-F
CE hard-negative mining
+ optional ranking loss

V6-G
OOF learned fusion

V6-H
target owner / reciprocal features
+ NONE threshold

V6-I
singleton/source-specific F0.5 decoder

V6-J
optional stronger reranker

V6-K
optional S2↔S3 triangle support
```

At every version:

```text
measure
save
compare
```

Do not continue using a component that reduces robust OOF macro F0.5.

# 50. REQUIRED ABLATION TABLE

Generate:

```text
experiment
pair_recall
full_entity_recall
oracle_f05
actual_macro_f05
singleton_f05
non_singleton_f05
S2_f05
S3_f05
FP
FN
blocker_FN
matcher_FN
mean_candidates
CE_pair_fraction
runtime
```

Write:

```text
artifacts/ablation_results.csv
```

# 51. REQUIRED DIAGNOSTIC: ORACLE GAP

For every experiment compute:

```python
oracle_gap =
    oracle_macro_f05 - actual_macro_f05
```

Interpret:

```text
large oracle gap
→ matcher/decoder problem

low oracle itself
→ retrieval/pruning problem
```

Print prominently.

# 52. REQUIRED DIAGNOSTIC: CE CONTRIBUTION

For routed validation pairs calculate:

```text
errors fixed by CE
new errors introduced by CE
FP fixed by CE
FN fixed by CE
```

Compare:

```text
structured only
scalar V5 blend
learned fusion
```

Do not claim CE helps unless final macro F0.5 improves.

# 53. REQUIRED DIAGNOSTIC: MODEL DISAGREEMENT

Bucket pairs by:

```text
structured high / CE high
structured high / CE low
structured low / CE high
structured low / CE low
```

Calculate true-match rate for each bucket.

Also break down disagreement by:

```text
numeric conflict
DBA
source
country
```

Use this to validate fusion behavior.

# 54. HIGH-PRECISION RULE LAYER

Only create deterministic accept/reject rules after measuring validation precision.

Potential accept candidates:

```text
exact distinctive name
+ exact/near-exact address
+ no numeric contradiction
```

Potential reject candidates:

```text
primary-number conflict
+ no strong supporting address evidence
+ generic name
```

A rule may be deployed only if:

```text
support is meaningful
AND
validation precision is near-perfect
AND
macro F0.5 improves
```

Do not hard-code intuition without measurement.

# 55. TARGET COLLISION AUDIT

After prediction:

```text
targets assigned to >1 S1
assignments affected
max target reuse
top reused target IDs
```

If target exclusivity was verified, final decoder should not produce multiple owners unless an explicit validated exception exists.

# 56. FINAL TRAINING

After architecture is chosen:

```text
1. freeze config from OOF/CV
2. regenerate full train candidates
3. include all GT positives
4. mine full hard negatives
5. train final structured models
6. train final CE
7. train final fusion using proper OOF methodology / final refit
8. freeze decoder thresholds
9. process full test
10. generate candidate_pairs.tsv
11. generate matching_results.tsv
```

Do not tune thresholds using test.

# 57. FINAL CANDIDATE FILE

`candidate_pairs.tsv` must represent:

```text
the final set of candidates actually entering the final matching system
```

If candidate undergoes structured model but is never CE-routed, it is still a final model candidate.

Every final match must be inside candidate list.

# 58. FINAL OUTPUT VALIDATION

Require:

```text
every test S1 exactly once
no duplicate S1 rows
only valid S2/S3 target IDs
no duplicate target inside row
matches subset of candidates
correct empty singleton encoding
```

Run official:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Require:

```text
PASS
```

# 59. REPRODUCIBILITY

Do not silently load stale model files.

Provide explicit commands:

```text
train
validate
predict
run-all
```

A `run-all` from raw challenge data must recreate outputs.

Version model artifacts by experiment/config hash.

# 60. README / DOCUMENTATION

Update only after final architecture is selected.

README must refer to final pipeline.

Pin exact dependency versions.

Document:

```text
normalization
retrieval channels
candidate oracle
structured features
CE architecture
hard-negative mining
ranking objective
fusion
owner logic
decoder
validation
France strategy
fair-play compliance
```

No placeholders.

# 61. STOPPING / CHAMPION RULE

Maintain current champion.

A candidate version replaces champion only if:

```text
robust OOF macro F0.5 improves
AND
no serious source/country collapse
AND
candidate oracle remains sufficient
AND
test inference is computationally feasible
```

Do not select using training score.

# 62. FINAL REPORT

Return exactly:

```text
1. V5 baseline validation macro F0.5
2. final validation macro F0.5
3. candidate pair recall
4. candidate full-entity recall
5. candidate oracle macro F0.5
6. oracle gap
7. singleton F0.5
8. non-singleton F0.5
9. S2 F0.5
10. S3 F0.5
11. FP count
12. FN count
13. blocker-caused FN
14. matcher-caused FN
15. CE-routed pair fraction
16. CE errors fixed / errors introduced
17. target collision statistics
18. selected thresholds
19. selected model configuration
20. complete ablation table
21. final test output statistics
22. official validator PASS/FAIL
23. exact reproduction commands
```

# 63. PRIORITY IF TIME RUNS SHORT

Mandatory:

```text
P0:
full-density validation
oracle ceiling
multi-field retrieval
stage1 recall safety-net
source preservation
normalization/DBA fixes
full-positive hard-negative training
risk-based CE routing
hard-negative CE training
OOF learned fusion
target-owner margins
F0.5 threshold tuning
valid test submission
```

Next:

```text
P1:
ranking loss
singleton model
source-conditioned LLR
calibration
```

Optional:

```text
P2:
stronger 568M reranker
dense retrieval
S2↔S3 triangle features
cardinality model
```

Do not spend time on P2 while oracle or OOF methodology remains incorrect.

# 64. CORE OBJECTIVE

V5:

```text
fixed-field retrieval
→ structured matcher
→ uncertainty-routed MiniLM
→ scalar logit blend
```

V6 must become:

```text
independent high-recall name/address/numeric retrieval
→ recall-safe structured pruning
→ source-aware structured matcher
→ risk-routed hard-negative-trained cross-encoder
→ OOF learned fusion
→ target-owner ranking/reciprocal evidence
→ source-aware singleton/F0.5 decoding
```

Do not simplify this architecture back to a single thresholded LightGBM or a standalone Transformer.

Implement, validate, ablate, retain measured improvements only.