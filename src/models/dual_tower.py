"""
Dual‑Tower training (no CLI flags).

* Reads:
    • data/processed/items.parquet      ← parent_asin, vector_id, title, description
    • data/embeddings/book_vecs.npy     ← fixed book vectors (pre‑computed)
    • data/processed/data_*.parquet     ← user events (user_id, parent_asin, timestamp)
* Builds user sequences ≥2 items, samples a random negative per row.
* Freezes the book‑tower embedding; trains a GRU User‑tower with triplet‑margin loss.
* Writes checkpoint → checkpoints/dualtower_best.ckpt

Run:
    python src/models/train_dualtower.py
"""
from __future__ import annotations

import glob
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# ── Config ───────────────────────────────────────────────────────────
ITEMS_PATH   = Path("data/processed/items.parquet")
BOOK_VECS    = Path("data/embeddings/book_vecs.npy")
SHARDS_GLOB  = "data/preprocessed/data_*.parquet"

SEQ_LEN_MAX  = 50
BATCH_SIZE   = 512
EPOCHS       = 3
GPUS         = 1          # 0 = CPU
LR           = 1e-4
MARGIN       = 0.2
PERSISTENT_WORKERS = True

# ── Dataset ──────────────────────────────────────────────────────────
class EventDataset(Dataset):
    """Returns (hist_ids, pos_id, neg_id) per sample."""
    def __init__(self, shards_glob: str, items_df: pd.DataFrame):
        asin2vid = items_df.set_index("parent_asin").vector_id.to_dict()
        self.max_id = max(asin2vid.values())

        sessions: dict[str, list[int]] = defaultdict(list)
        for fp in glob.glob(shards_glob):
            df = pd.read_parquet(fp, columns=["user_id", "parent_asin", "timestamp"])
            df = df.sort_values(["user_id", "timestamp"], ignore_index=True)
            for uid, grp in df.groupby("user_id"):
                sessions[uid].extend(
                    asin2vid[a] for a in grp.parent_asin if a in asin2vid
                )

        self.seqs = [v[-SEQ_LEN_MAX:] for v in sessions.values() if len(v) >= 2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        hist = torch.tensor(seq[:-1], dtype=torch.long)
        pos  = torch.tensor(seq[-1],  dtype=torch.long)
        neg = pos
        while neg.item() == pos.item():
            neg = torch.tensor(random.randint(0, self.max_id), dtype=torch.long)
        return hist, pos, neg

# ── Collate fn: pad histories ────────────────────────────────────────
PAD_IDX = 0

def collate(batch):
    h, p, n = zip(*batch)
    hist = nn.utils.rnn.pad_sequence(h, padding_value=PAD_IDX, batch_first=True)
    return hist, torch.stack(p), torch.stack(n)

# ── Model components ────────────────────────────────────────────────
class UserTower(nn.Module):
    def __init__(self, item_emb: nn.Embedding, dim: int):
        super().__init__()
        self.item_emb = item_emb
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hist_ids):                  # (B,T)
        x, _ = self.gru(self.item_emb(hist_ids))  # (B,T,D) → (B,D)
        z = self.proj(x[:, -1])                   # last hidden
        return torch.nn.functional.normalize(z, dim=-1)

class DualTowerLit(pl.LightningModule):
    def __init__(self, book_vecs: np.ndarray):
        super().__init__()
        dim = book_vecs.shape[1]
        emb = nn.Embedding.from_pretrained(torch.tensor(book_vecs), freeze=True)
        self.item_emb   = emb
        self.user_tower = UserTower(emb, dim)
        self.loss_fn    = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - (x * y).sum(-1), margin=MARGIN)
        # log only scalars
        self.save_hyperparameters({"dim": dim, "margin": MARGIN}, ignore=["book_vecs"])

    def training_step(self, batch, _):
        hist, pos_id, neg_id = batch
        u   = self.user_tower(hist)
        pos = self.item_emb(pos_id)
        neg = self.item_emb(neg_id)
        loss = self.loss_fn(u, pos, neg)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.user_tower.parameters(), lr=LR)

# ── Main script ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # artefacts
    items_df  = pd.read_parquet(ITEMS_PATH)
    book_vecs = np.load(BOOK_VECS)

    ds = EventDataset(SHARDS_GLOB, items_df)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=collate,
    )

    model = DualTowerLit(book_vecs)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if GPUS else "cpu",
        devices=GPUS or 1,
        log_every_n_steps=50,
    )
    trainer.fit(model, dl)

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    trainer.save_checkpoint(ckpt_dir / "dualtower_best.ckpt")
    print("checkpoint saved to", ckpt_dir / "dualtower_best.ckpt")
