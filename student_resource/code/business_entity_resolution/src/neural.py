"""Neural components (optional, --use-emb / --use-ce). Needs a CUDA GPU for full-size data.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (Apache-2.0, ~118M params)
  * bi-encoder  : embeds RAW business names (native scripts kept - the model is multilingual,
                  so Kannada/Hindi/French names land near their Latin counterparts).
                  Embeddings are computed once per unique string, cached on disk as fp16.
  * cross-encoder: the same backbone + a 1-logit head, fine-tuned on hard candidate pairs
                  "name ; address" x "name ; address". Applied only to the uncertain zone of
                  stage 2, blended as  logit + w * ce_logit  (w tuned on val macro F0.5).
Pretrained weights are downloaded from the Hugging Face hub; no business data leaves the machine.
"""
import hashlib
import math
import os

import numpy as np
import pandas as pd

_ST = {}


def _device(cfg):
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if cfg.nn_allow_cpu:
        return "cpu"
    raise RuntimeError("No CUDA GPU visible to torch. Install the CUDA build of torch "
                       "(see README) or pass --nn-allow-cpu true for tiny tests.")


def _st_model(cfg):
    if cfg.nn_model not in _ST:
        from sentence_transformers import SentenceTransformer
        dev = _device(cfg)
        m = SentenceTransformer(cfg.nn_model, device=dev)
        if dev == "cuda":
            m = m.half()
        m.max_seq_length = 48
        _ST[cfg.nn_model] = m
    return _ST[cfg.nn_model]


class NameEmbeddings:
    """Unique-string fp16 embeddings in a disk memmap + per-record codes."""

    def __init__(self, texts, cfg, tag, log=print):
        texts = pd.Series(texts, dtype=str).fillna("").str.strip()
        self.codes, uniq = pd.factorize(texts)
        uniq = list(uniq)
        key = hashlib.md5(("\n".join(uniq[:1000]) + str(len(uniq)) + cfg.nn_model).encode()).hexdigest()[:10]
        os.makedirs(os.path.join(cfg.model_dir, "emb_cache"), exist_ok=True)
        path = os.path.join(cfg.model_dir, "emb_cache", f"{tag}_{key}.npy")
        if not os.path.exists(path):
            m = _st_model(cfg)
            dim = m.get_sentence_embedding_dimension()
            tmp = path + ".tmp.npy"
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(uniq), dim))
            step = 200_000
            for i in range(0, len(uniq), step):
                out[i:i + step] = m.encode(uniq[i:i + step], batch_size=cfg.emb_batch,
                                           normalize_embeddings=True, convert_to_numpy=True,
                                           show_progress_bar=False).astype(np.float16)
                log(f"    embedded {min(i + step, len(uniq)):,}/{len(uniq):,} unique names ({tag})")
            out.flush()
            del out
            os.replace(tmp, path)
        self.U = np.load(path, mmap_mode="r")

    def rows(self, idx):
        return np.asarray(self.U[self.codes[idx]], dtype=np.float32)


def emb_cos(E1, EQ, s, q, chunk=1_000_000):
    out = np.empty(len(s), dtype=np.float32)
    for i in range(0, len(s), chunk):
        j = slice(i, i + chunk)
        out[j] = np.einsum("ij,ij->i", E1.rows(s[j]), EQ.rows(q[j]))
    return out


# ------------------------------------------------------------------ cross-encoder
def pair_texts(P1, PQ, s, q):
    return P1.raw.values[s].tolist(), PQ.raw.values[q].tolist()


def train_cross_encoder(ta, tb, y, cfg, out_dir, log=print):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    dev = _device(cfg)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.nn_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.nn_model, num_labels=1).to(dev)
    model.train()
    n, bs = len(y), cfg.ce_batch
    steps = cfg.ce_epochs * math.ceil(n / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ce_lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(cfg.seed)
    y = np.asarray(y, dtype=np.float32)
    step = 0
    for ep in range(cfg.ce_epochs):
        perm = rng.permutation(n)
        run = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            enc = tok([ta[k] for k in idx], [tb[k] for k in idx], truncation=True,
                      max_length=cfg.ce_max_len, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), torch.from_numpy(y[idx]).to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0 or step == steps:
                log(f"    cross-encoder epoch {ep + 1} step {step}/{steps} loss~{run:.4f}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


_CE = {}


def ce_score(ta, tb, cfg, model_dir, log=print):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if len(ta) == 0:
        return np.zeros(0, dtype=np.float32)
    dev = _device(cfg)
    if model_dir not in _CE:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
        if dev == "cuda":
            model = model.half()
        _CE[model_dir] = (tok, model)
    tok, model = _CE[model_dir]
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)], kind="stable")  # length-bucketing
    out = np.empty(len(ta), dtype=np.float32)
    bs = cfg.ce_batch * 8
    with torch.inference_mode():
        for n_done, i in enumerate(range(0, len(order), bs)):
            idx = order[i:i + bs]
            enc = tok([ta[k] for k in idx], [tb[k] for k in idx], truncation=True,
                      max_length=cfg.ce_max_len, padding=True, return_tensors="pt").to(dev)
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if n_done % 400 == 0:
                log(f"    cross-encoder scored {min(i + bs, len(order)):,}/{len(order):,}")
    return out
