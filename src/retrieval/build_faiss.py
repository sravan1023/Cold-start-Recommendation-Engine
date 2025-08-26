'''
Build a FAISS index from book embeddings for fast similarity search.

Usage:
    python src/retrieval/build_faiss.py
'''
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import faiss

# ── Configuration ─────────────────────────────────────────────────────
EMB_PATH     = Path("data/embeddings/book_vecs.npy")
ITEMS_PATH   = Path("data/processed/items.parquet")
INDEX_PATH   = Path("data/faiss/index_hnsw.bin")
MAPPING_PATH = Path("data/faiss/index_mapping.pkl")

# HNSW parameters
USE_HNSW         = True
HNSW_M           = 32
EF_CONSTRUCTION  = 200

# Number of nearest neighbors to retrieve at query time (example)
QUERY_K = 10
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1) Load embeddings and item mappings
    print(f"Loading embeddings from {EMB_PATH}")
    vecs = np.load(EMB_PATH)
    print(f"Loading item index from {ITEMS_PATH}")
    items = pd.read_parquet(ITEMS_PATH)

    assert len(vecs) == len(items), (
        f"Mismatch: {len(vecs)} vectors vs {len(items)} items"
    )
    d = vecs.shape[1]
    print(f"Number of items: {len(vecs)}, dimension: {d}")

    # 2) Build the FAISS index
    if USE_HNSW:
        index = faiss.IndexHNSWFlat(d, HNSW_M)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        print(f"Using HNSWFlat index: M={HNSW_M}, efConstruction={EF_CONSTRUCTION}")
    else:
        # Normalize for cosine similarity
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(d)
        print("Using Flat IP index (cosine similarity after normalization)")

    print("Adding vectors to the index...")
    index.add(vecs)
    print(f"Total vectors indexed: {index.ntotal}")

    # Ensure output dir exists
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 3) Save the FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index written to {INDEX_PATH}")

    # 4) Save vector_id -> parent_asin mapping
    mapping = items.set_index('vector_id')['parent_asin'].to_dict()
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Mapping saved to {MAPPING_PATH}")

    # Optional: test query
    # D, I = index.search(vecs[:1], QUERY_K)
    # print("Sample query results:", [items.iloc[i]['parent_asin'] for i in I[0]])


if __name__ == '__main__':
    main()
