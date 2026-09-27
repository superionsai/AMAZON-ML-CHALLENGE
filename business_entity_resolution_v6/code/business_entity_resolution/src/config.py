"""Pipeline configuration for Business Entity Resolution V6.

Optimized to reach >=0.99 macro F0.5 following IDEA.md.
Key upgrades:
- Multi-channel inclusive blocking (Joint TF-IDF + Name Char n-grams + Address Word tokens + Expanded Keys)
- Stage-1 pruning safety net (preserves top query neighbours, exact skeleton matches, and strong anchors)
- Rich pairwise features (numeric contradiction, prefix/sort/acronym matches, cross-field interactions)
- Dedicated singleton preservation gate + metric-calibrated expected-F0.5 set decoder
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    data_dir: str = "dataset"
    out_dir: str = "output"
    model_dir: str = "models"
    seed: int = 42

    # training sample
    train_s1_frac: float = 0.4      # 40% sample gives richer training data and harder negative mining
    val_frac: float = 0.2

    # blocking upper bounds (stage-1 model prunes below these)
    knn_k: int = 25                # max S1 neighbours retrieved in joint TF-IDF space
    knn_k_name: int = 15           # dedicated name char n-gram channel (retrieves missing address matches)
    knn_k_addr: int = 15           # dedicated address word channel (retrieves DBA / brand alias matches)
    w_name: float = 0.6            # name vs address weight in the joint TF-IDF space
    knn_max_df: float = 0.005      # retrieval ignores features in > this share of S1 records
    key_cap_s1: int = 200          # raised from 100 to prevent premature key block drops
    key_cap_q: int = 2000          # raised from 1000
    query_chunk: int = 200_000     # S2/S3 records processed per blocking chunk

    # policy: fraction of blocked true pairs the stage-1 pruning must keep
    prune_recall: float = 0.999    # raised from 0.998 to minimize false pruning
    safety_net: bool = True        # stage-1 safety net: never prune top query neighbors or exact keys

    # decision rule / singleton preservation
    singleton_gate: bool = True    # dedicated singleton confidence threshold

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
