"""
FastAPI inference server for the cold-start book recommender
-----------------------------------------------------------
Run locally (inside your venv):

    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

The same file works inside the Docker image because /app exists there.
"""

import os, time, pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.serve.blend import blend
from src.models.dual_tower import DualTowerLit   # import path matches file

# ---------------------------------------------------------------------
# Resolve data folder
# • In Docker → /app
# • Local run → repo root two levels up from this file
# ---------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/app"))
if not DATA_DIR.exists():       # fallback for local dev
    DATA_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Load artefacts once at startup
# ---------------------------------------------------------------------
print("Loading artefacts from", DATA_DIR)

book_vecs = np.load(DATA_DIR / "data/embeddings/book_vecs_dualtower.npy")

index = faiss.read_index(str(DATA_DIR / "data/faiss/index_dualtower_hnsw.bin"))
with open(DATA_DIR / "data/faiss/index_mapping.pkl", "rb") as f:
    id2asin: dict[int, str] = pickle.load(f)
asin2id = {v: k for k, v in id2asin.items()}

model = DualTowerLit.load_from_checkpoint(
    DATA_DIR / "checkpoints/dualtower_best.ckpt",
    book_vecs=book_vecs,
).eval().cpu()          # switch to .cuda() if you have GPU & torch-cuda

# popularity list (ASINs only)
pop_overall = list(np.loadtxt(
    DATA_DIR / "data/popularity/top_overall.parquet",
    dtype=str,
    delimiter=",",
    usecols=0
))

print("Ready — API will serve on /recommend")

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Book Recommender")

class Query(BaseModel):
    user_id: str | None = None
    history: List[str] = []
    k: int = 20

# helper
def faiss_query(vec: np.ndarray, topk: int) -> List[str]:
    D, I = index.search(vec.astype("float32"), topk)
    return [id2asin[i] for i in I[0]]

# endpoint
@app.post("/recommend")
def recommend(q: Query):
    t0 = time.time()
    K = q.k

    # rail 1: popularity
    pop_ids = pop_overall[: K * 5]

    # rail 2 + 3 depend on history
    if q.history:
        # ---- build user vector from history
        hist_valid = [asin2id[a] for a in q.history if a in asin2id]
        if not hist_valid:
            # fall back to cold-start if no ids match
            faiss_ids = faiss_query(book_vecs[0:1], K * 10)
            dual_ids: List[Tuple[str, float]] = []
        else:
            hist_ids = torch.tensor([hist_valid[-50:]])  # (1,T)
            user_vec = model.user_tower(hist_ids).detach().cpu().numpy()
            # rail 2: FAISS recall
            faiss_ids = faiss_query(user_vec, K * 10)
            # rail 3: Dual-tower score for each FAISS candidate
            dual_ids = [(a, float(user_vec @ book_vecs[asin2id[a]]))
                        for a in faiss_ids]
    else:
        # cold-start: query FAISS with a dummy seed (first vector)
        faiss_ids = faiss_query(book_vecs[0:1], K * 10)
        dual_ids = []

    slate = blend(pop_ids, faiss_ids, dual_ids, k=K)
    return {
        "slate": slate,
        "latency_ms": int((time.time() - t0) * 1000)
    }