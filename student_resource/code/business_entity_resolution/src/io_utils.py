import os

import pandas as pd


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_split(data_dir, split):
    rd = lambda n: read_tsv(os.path.join(data_dir, split, f"{split}_{n}.tsv"))
    s1, s2, s3 = rd("source1"), rd("source2"), rd("source3")
    gt = rd("ground_truth") if split == "train" else None
    return s1, pd.concat([s2, s3], ignore_index=True), gt


def truth_pairs(gt):
    """Ground truth -> DataFrame(s1_id, q_id)."""
    t = gt.assign(q_id=gt.matched_entity_ids.str.split(",")).explode("q_id")
    t = t[t.q_id.notna() & (t.q_id != "")]
    return pd.DataFrame({"s1_id": t.source1_entity_id.values, "q_id": t.q_id.values})


def write_id_lists(path, header_col, s1_ids, pairs):
    """pairs: DataFrame(s1_id, q_id). One row per S1 id, sorted unique ids, tab-separated."""
    grouped = (pairs.drop_duplicates().sort_values(["s1_id", "q_id"])
               .groupby("s1_id").q_id.agg(",".join).to_dict())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{grouped.get(sid, '')}\n")
