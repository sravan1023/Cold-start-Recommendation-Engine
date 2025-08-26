"""
Evaluate Recall@K (10, 20, 50) for the cold-start blender.

Run from repo root:
    python src/offline/eval_recall.py
"""

from __future__ import annotations
import sys, glob, pickle, faiss, numpy as np, pandas as pd, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── make `src` importable no matter where we run ----------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve.blend import blend
from src.models.dual_tower import DualTowerLit

# ── paths -------------------------------------------------------------
SHARDS_GLOB = ROOT / "data/preprocessed/data_*.parquet"
VEC_NPY     = ROOT / "data/embeddings/book_vecs_dualtower.npy"
INDEX_BIN   = ROOT / "data/faiss/index_hnsw.bin"
MAP_PKL     = ROOT / "data/faiss/index_mapping.pkl"
CKPT_PATH   = ROOT / "checkpoints/dualtower_best.ckpt"
POP_PARQ    = ROOT / "data/popularity/top_overall.parquet"

TOP_KS = (10, 20, 50)

# ── load artefacts ----------------------------------------------------
print("Loading vectors, FAISS index, mapping …")
book_vecs = np.load(VEC_NPY)
index     = faiss.read_index(str(INDEX_BIN))

id2asin   = pickle.load(open(MAP_PKL, "rb"))
asin2id   = {v: k for k, v in id2asin.items()}

assert len(book_vecs) == len(asin2id), "vector / mapping size mismatch"

dual = DualTowerLit.load_from_checkpoint(
    CKPT_PATH, book_vecs=book_vecs
).eval().cpu()

pop_overall = pd.read_parquet(POP_PARQ).parent_asin.tolist()
pop_ids     = pop_overall[:20_000]                    # BIG popularity rail

seed_vec = book_vecs[[asin2id[a] for a in pop_overall[:100]]].mean(0, keepdims=True)

def faiss_query(vec: np.ndarray, k: int = 2_000) -> list[str]:
    _, I = index.search(vec.astype("float32"), k)
    return [id2asin[i] for i in I[0] if i != -1]

# ── build test set ----------------------------------------------------
print(" Building user sequences …")
sessions: dict[str, list[str]] = defaultdict(list)
for fp in tqdm(sorted(glob.glob(str(SHARDS_GLOB))), unit="shard"):
    df = pd.read_parquet(fp, columns=["user_id", "parent_asin", "timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    for uid, grp in df.groupby("user_id"):
        sessions[uid].extend(grp.parent_asin.tolist())

tests = [(seq[:-1], seq[-1]) for seq in sessions.values() if len(seq) >= 3]
print(f"Test users: {len(tests):,}")

# quick mapping coverage check
covered = sum(gt in asin2id for _, gt in tests)
print(f"Ground-truth coverage in mapping: {covered/len(tests):.1%}")

# ── evaluate recall ---------------------------------------------------
hits = {k: 0 for k in TOP_KS}

for hist, gt in tqdm(tests, unit="user"):
    # rail 1: popularity (already pop_ids)
    valid_ids = [asin2id[a] for a in hist if a in asin2id][-50:]

    # rails 2 & 3
    if valid_ids:
        user_vec = dual.user_tower(torch.tensor([valid_ids])).detach().numpy()
        faiss_ids = faiss_query(user_vec, 2_000)
        dual_ids  = [(a, float(user_vec @ book_vecs[asin2id[a]]))
                     for a in faiss_ids]
    else:
        faiss_ids = faiss_query(seed_vec, 2_000)
        dual_ids  = []

    slate = blend(pop_ids, faiss_ids, dual_ids, k=max(TOP_KS))

    for k in TOP_KS:
        if gt in slate[:k]:
            hits[k] += 1

# ── report ------------------------------------------------------------
print("\nRecall@K:")
for k in TOP_KS:
    r = hits[k] / len(tests)
    print(f"  Recall@{k:<2d}: {r:.3%}")

