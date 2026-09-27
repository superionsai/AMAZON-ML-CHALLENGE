# Business Entity Resolution - pipeline

Run everything from the `student_resource/` directory (the one containing `dataset/`).

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run.py train     # models/ + learned thresholds
python code/business_entity_resolution/src/run.py predict   # output/*.tsv
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Useful flags (all in `src/config.py`): `--train-s1-frac 0.05` for a fast smoke test,
`--n-threads 8`, `--query-chunk 200000` if RAM is tight, `--knn-k`, `--w-name`.

## Stages
| step | file | what |
|---|---|---|
| normalise | `normalize.py` | transliteration (anyascii), legal-suffix/DBA/domain/honorific stripping, phonetic consonant skeleton, address abbreviation + state/city alias maps |
| block | `blocking.py` | per country: joint TF-IDF kNN (name skeleton char n-grams + address words) UNION six deterministic rarity keys |
| stage 1 | `features.py`, `run.py` | cheap LightGBM -> p1; learned cut tau keeps `prune_recall` of blocked true pairs. The surviving pairs are written to `candidate_pairs.tsv` and are exactly what stage 2 scores |
| stage 2 | `features.py`, `run.py` | LightGBM on ~60 features (fuzzy name/address, house-number edit distance, context ranks/margins, duplicate support) |
| decide | `optimize.py` | one-to-one assignment, per-S1 expected-F0.5 subset rule, calibration (T, b, c) fitted by coordinate ascent + golden-section on exact validation macro F0.5 |

## Neural extras (`--use-emb true --use-ce true`)
`neural.py`, model `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M params, well under 8B):
* **bi-encoder**: embeds raw names (native scripts kept); `emb_cos` + its per-query/per-S1 rank/margins feed stages 1 and 2. Embeddings are cached per unique string in `models*/emb_cache/`.
* **cross-encoder**: same backbone + 1-logit head fine-tuned on uncertainty-weighted training pairs, applied to pairs with |stage-2 logit| < `ce_margin`; final score `logit + w*ce`, with `w` tuned jointly with (T, b, c) on validation macro F0.5.

Needs the CUDA build of torch. `overnight.ps1` chains train/predict/validate for the neural model and a non-neural fallback.

No external data or APIs are used (pretrained weights are downloaded once from the Hugging Face hub; no business data is sent anywhere). Country is never a model feature, so unseen countries (France) run through the same code path.
