"""
Blend‑demo – run completely standalone.

• Works when executed directly (`python src/serve/demo.py`)
  because it appends the repo root to `sys.path` automatically.
• Prints 20 recommended ASINs with their titles.
"""

from __future__ import annotations
import sys, math, pickle, faiss, numpy as np, pandas as pd
from pathlib import Path

# ── ensure src/ is importable -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # repo root two levels up
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve.blend import blend    # now import works

# ── load artefacts ------------------------------------------------------------
items_path   = ROOT / "data/processed/items.parquet"
vec_path     = ROOT / "data/embeddings/book_vecs_dualtower.npy"
index_path   = ROOT / "data/faiss/index_hnsw.bin"
mapping_path = ROOT / "data/faiss/index_mapping.pkl"

print("Loading items …", items_path)
items = pd.read_parquet(items_path, columns=["parent_asin", "title"])
asin2title = dict(zip(items.parent_asin, items.title))

print("Loading vectors & FAISS index …")
book_vecs = np.load(vec_path)
index     = faiss.read_index(str(index_path))
with open(mapping_path, "rb") as f:
    id2asin = pickle.load(f)
asin2id = {v: k for k, v in id2asin.items()}

# helper
def faiss_query(vec: np.ndarray, k: int = 300) -> list[str]:
    _, I = index.search(vec.astype("float32"), k)
    return [id2asin[i] for i in I[0]]

# ── build candidate rails -----------------------------------------------------
pop_ids   = items.parent_asin.head(500).tolist()          # popularity stub
faiss_ids = faiss_query(book_vecs[0:1], 300)              # cold‑start seed
dual_ids  = []                                            # no user vector yet

slate = blend(pop_ids, faiss_ids, dual_ids, k=20)

print("\nRecommended (ASIN → title):")
for rank, asin in enumerate(slate, 1):
    print(f"{rank:2d}. {asin} → {asin2title.get(asin, '<missing title>')}")
