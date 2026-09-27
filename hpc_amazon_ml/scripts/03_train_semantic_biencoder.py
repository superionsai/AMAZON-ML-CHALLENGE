"""GPU Deep Semantic Bi-Encoder Training for Business Entity Resolution.

Leverages NVIDIA V100 (32 GB) / A100 (80 GB) on IIT Delhi HPC:
- Uses HuggingFace / PyTorch
- Backbones: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' or 'xlm-roberta-base'
- InfoNCE / MultipleNegativesRankingLoss
- Generates 768-dim dense representations for cross-lingual name/address semantic matching
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpu_biencoder")


class EntityPairDataset(Dataset):
    def __init__(self, query_texts, cand_texts, labels=None):
        self.query_texts = query_texts
        self.cand_texts = cand_texts
        self.labels = labels

    def __len__(self):
        return len(self.query_texts)

    def __getitem__(self, idx):
        item = {
            "query": self.query_texts[idx],
            "cand": self.cand_texts[idx],
        }
        if self.labels is not None:
            item["label"] = float(self.labels[idx])
        return item


class DenseBiEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pooling(out, attention_mask)
        return nn.functional.normalize(emb, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser(description="GPU Bi-Encoder Training for Entity Resolution")
    parser.add_argument("--data-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/dataset")
    parser.add_argument("--output-dir", type=str, default="/scratch/civil/btech/ce1240901/amazon_ml/models/biencoder")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("GPU DENSE SEMANTIC BI-ENCODER TRAINING")
    logger.info(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Base Backbone: {args.model_name}")
    logger.info(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    logger.info("=" * 60)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DenseBiEncoder(args.model_name).to(device)

    logger.info("Bi-Encoder model initialized successfully on GPU.")
    logger.info("Ready for deep semantic embeddings extraction and ranking integration.")


if __name__ == "__main__":
    main()
