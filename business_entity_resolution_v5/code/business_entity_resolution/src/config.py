"""Pipeline configuration.

Only *upper bounds* and policy knobs live here. Everything that decides the final
candidate set and the final matches (pruning threshold, calibration, per-entity
cut-off) is learned on the validation split by src/optimize.py.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    data_dir: str = "dataset"
    out_dir: str = "output"
    model_dir: str = "models"
    seed: int = 42

    # training sample (keeps laptop RAM sane; ratios of matches/distractors preserved)
    train_s1_frac: float = 0.3
    val_frac: float = 0.2

    # blocking upper bounds (stage-1 model prunes below these)
    knn_k: int = 20               # max S1 neighbours retrieved per S2/S3 record
    w_name: float = 0.6           # name vs address weight in the joint TF-IDF space
    knn_max_df: float = 0.005     # retrieval ignores features in > this share of S1 records
                                  # (scale-free, so train sample and full test behave alike)
    key_cap_s1: int = 100         # deterministic-key blocks with more S1 records are skipped
    key_cap_q: int = 1000         # ... or with more S2/S3 records
    query_chunk: int = 200_000    # S2/S3 records processed per blocking chunk

    # policy: fraction of blocked true pairs the stage-1 pruning must keep
    prune_recall: float = 0.998

    n_threads: int = 8

    # ---------------- neural (optional; needs CUDA torch)
    use_emb: bool = False          # multilingual bi-encoder name-embedding features
    use_ce: bool = False           # fine-tuned cross-encoder on the stage-2 uncertain zone
    nn_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0
    nn_allow_cpu: bool = False
    emb_batch: int = 1024
    ce_batch: int = 64
    ce_lr: float = 3e-5
    ce_epochs: int = 1
    ce_max_len: int = 96
    ce_train_pairs: int = 400_000
    ce_margin: float = 3.0         # rescore pairs with |stage-2 logit| below this
    ce_max_rows: int = 4_000_000   # per-country cap on rescored test pairs (most uncertain first)

    def dict(self):
        return asdict(self)
