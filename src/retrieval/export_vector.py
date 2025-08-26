'''
Export (or just copy) item vectors from the dual‑tower checkpoint.

If your Book‑tower embeddings were **frozen** during training, the
vectors are identical to the original `book_vecs.npy`, but this script
still writes them out under a new name so downstream code can depend
only on the checkpoint.

Outputs
-------
• data/embeddings/book_vecs_dualtower.npy   –  (N_items, D) float32
• data/processed/items.parquet  (unchanged, kept for mapping)

Run:
    python src/retrieval/export_vectors.py
'''
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────
CKPT_PATH  = Path("checkpoints/dualtower_best.ckpt")
ITEMS_PATH = Path("data/processed/items.parquet")   # for sanity check
OUT_NPY    = Path("data/embeddings/book_vecs_dualtower.npy")
# --------------------------------------------------------------------

def main() -> None:
    # 1) load checkpoint weights (cpu is fine)
    print("Loading checkpoint →", CKPT_PATH)
    state = torch.load(CKPT_PATH, map_location="cpu")

    # item_emb.weight is saved under 'item_emb.weight' by Lightning
    if "item_emb.weight" not in state["state_dict"]:
        raise KeyError("item_emb.weight not found in checkpoint."
                        " Did you rename or strip the layer?")

    weight = state["state_dict"]["item_emb.weight"].cpu().numpy().astype("float32")
    print("Vector matrix shape:", weight.shape)

    # 2) optional sanity check vs items.parquet
    if ITEMS_PATH.exists():
        n_items = len(pd.read_parquet(ITEMS_PATH))
        assert n_items == weight.shape[0], (
            f"Mismatch: {n_items} items in mapping vs {weight.shape[0]} vectors")

    # 3) save to .npy
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, weight)
    print("Saved fine‑tuned item vectors ", OUT_NPY)


if __name__ == "__main__":
    main()
