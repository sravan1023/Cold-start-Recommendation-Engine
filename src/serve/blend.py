"""
Serve‑time blender for cold‑start recommendation.

Inputs (already computed elsewhere)
----------------------------------
• pop_list    list[str]           (most‑popular ASINs)
• faiss_list  list[str]           (content similarity from FAISS)
• dual_list   list[tuple[str, float]]  (ASIN, dual‑tower score)

The blender:
1. Normalises dual‑tower scores to [0, 1]
2. Assigns a weight to each **source rail**
   - pop_weight   (default 0.20)
   - faiss_weight (default 0.30)
   - dual_weight  (default 0.50)
3. Produces a merged slate of unique ASINs ranked by combined score.

Usage
-----
>>> slate = blend(pop_list, faiss_list, dual_list, k=20)
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd

# ── Weights & constants ──────────────────────────────────────────────
POP_WEIGHT   = 0.20
FAISS_WEIGHT = 0.30
DUAL_WEIGHT  = 0.50

DEFAULT_K    = 20

# ── Optional title lookup (for demo printing) ────────────────────────
ITEMS_PATH = Path(__file__).resolve().parents[2] / "data/processed/items.parquet"
if ITEMS_PATH.exists():
    _items_df = pd.read_parquet(ITEMS_PATH, columns=["parent_asin", "title"])
    ASIN2TITLE: dict[str, str] = dict(zip(_items_df.parent_asin, _items_df.title))
else:
    ASIN2TITLE = {}

# --------------------------------------------------------------------

def _rankify(seq: List[str]) -> Dict[str, float]:
    """Turn an ordered list into inverse‑rank scores in [0,1]."""
    return {asin: 1.0 / (i + 1) for i, asin in enumerate(seq)}


def blend(
    pop_list:   List[str],
    faiss_list: List[str],
    dual_list:  List[Tuple[str, float]],  # (asin, score)
    k: int = DEFAULT_K,
) -> List[str]:
    """Merge three candidate rails → top‑k unique ASIN slate."""

    # 1) normalise dual scores to 0‑1
    if dual_list:
        d_scores = [s for _, s in dual_list]
        lo, hi = min(d_scores), max(d_scores)
        span = hi - lo or 1e-6
        dual_norm = {a: (s - lo) / span for a, s in dual_list}
    else:
        dual_norm = {}

    # 2) convert pop & faiss to inverse‑rank scores
    pop_rank   = _rankify(pop_list)
    faiss_rank = _rankify(faiss_list)

    # 3) weighted sum per ASIN
    scores: Dict[str, float] = {}
    for asin, s in dual_norm.items():
        scores[asin] = scores.get(asin, 0) + DUAL_WEIGHT * s
    for asin, s in faiss_rank.items():
        scores[asin] = scores.get(asin, 0) + FAISS_WEIGHT * s
    for asin, s in pop_rank.items():
        scores[asin] = scores.get(asin, 0) + POP_WEIGHT * s

    # 4) sort & return unique top‑k
    slate = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [asin for asin, _ in slate]


# ── Minimal demo ------------------------------------------------------
if __name__ == "__main__":
    # fake rails
    pop   = [f"B{1000+i}" for i in range(100)]
    faiss = [f"B{2000+i}" for i in range(50)] + pop[:50]      # some overlap
    dual  = [(f"B{1000+i}", math.exp(-i/10)) for i in range(75)]

    slate = blend(pop, faiss, dual, k=20)

    print("Recommended (ASIN :Title):")
    for asin in slate:
        title = ASIN2TITLE.get(asin, "<title unknown>")
        print(f"  {asin}  : {title}")

