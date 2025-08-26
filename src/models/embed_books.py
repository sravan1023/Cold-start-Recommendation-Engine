# ── src/models/embed_books.py ─────────────────────────────────────────
"""
Embed every unique book (parent_asin) into a fixed-length vector.

Input  shards :  data/processed/data_*.parquet
Outputs       :
    • data/processed/items.parquet       (parent_asin ↔ vector_id + text)
    • data/embeddings/book_vecs.npy      (N_items, D) float32

Run:
    python src/models/embed_books.py
"""

from pathlib import Path
import glob, numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────
SHARDS_GLOB = "data/preprocessed/data_*.parquet"
ITEMS_OUT   = Path("data/processed/items.parquet")
VECS_OUT    = Path("data/embeddings/book_vecs.npy")

MODEL_NAME  = "all-MiniLM-L6-v2"   # switch to your fine-tuned model later
BATCH_SIZE  = 256                  # GPU: 256–512, CPU: 64–128
# ──────────────────────────────────────────────────────────────────────


def load_unique_items() -> pd.DataFrame:
    """Concatenate shards, drop duplicate books."""
    frames = []
    for fp in tqdm(sorted(glob.glob(SHARDS_GLOB)),
                   desc="Loading shards", unit="shard", colour="cyan"):
        frames.append(pd.read_parquet(fp,
                       columns=["parent_asin", "title", "description"]))
    items = (pd.concat(frames, ignore_index=True)
               .drop_duplicates("parent_asin")
               .reset_index(drop=True))
    items["vector_id"] = np.arange(len(items), dtype=np.int32)
    print("Unique books:", len(items))
    return items


def build_text_column(df: pd.DataFrame) -> list[str]:
    """title + description (NaNs handled) → list[str]"""
    return (df["title"].fillna("") + ". " +
            df["description"].fillna("")).tolist()


def main() -> None:
    # 1. Load & deduplicate items
    items = load_unique_items()

    # 2. Prepare texts
    texts = build_text_column(items)

    # 3. Embed with Sentence-Transformers
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm
    ).astype("float32")

    # 4. Save artefacts
    VECS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECS_OUT, vecs)
    items.to_parquet(ITEMS_OUT, index=False)

    print(f"Saved vectors → {VECS_OUT}  (shape {vecs.shape})")
    print(f"Saved mapping → {ITEMS_OUT}")


if __name__ == "__main__":
    main()
