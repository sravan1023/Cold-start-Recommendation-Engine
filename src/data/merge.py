# ── src/data/clean_merge.py ───────────────────────────────────────────
"""
Clean + merge Amazon-Books review & metadata shards, then save as parquet
shards for fast downstream loading.

Usage (from repo root, venv active):
    python -m src.data.clean_merge
"""

from pathlib import Path
import glob
import pandas as pd


# ── CONFIG ────────────────────────────────────────────────────────────
REV_GLOB      = "data/interim/reviews_*.parquet"
META_GLOB     = "data/interim/meta_*.parquet"
OUT_DIR       = Path("data/pre_cleaned")
PREFIX        = "data"        # final file prefix
SHARD_SIZE    = 250_000         # rows per output shard
MIN_LEN       = 15              # keep reviews with ≥ 15 characters
MAX_REV_USER  = 500             # cap reviews per user (None = no cap)
# ──────────────────────────────────────────────────────────────────────


def load_glob(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"[ERR] No parquet files match: {pattern}")
    print(f"Loading {len(files)} shards from {pattern} …")
    return pd.concat(pd.read_parquet(f) for f in files)


def write_shards(df: pd.DataFrame, out_dir: Path, prefix: str, size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(df)
    n_shards = (n_rows + size - 1) // size
    for i in range(n_shards):
        start, end = i * size, min((i + 1) * size, n_rows)
        shard = df.iloc[start:end]
        fname = out_dir / f"{prefix}_{i:05d}.parquet"
        shard.to_parquet(fname, index=False)
        print(f"wrote rows {start:,}-{end-1:,} -> {fname.name}")


def main() -> None:
    # 1) Load all review & meta shards
    reviews = load_glob(REV_GLOB)
    meta = load_glob(META_GLOB)[["parent_asin", "title", "description"]]

    print("Original reviews shape :", reviews.shape)

    # 2) Minimal cleaning
    reviews = reviews[reviews["text"].str.len() >= MIN_LEN]
    if MAX_REV_USER:
        reviews = (
            reviews.sort_values("timestamp")
                   .groupby("user_id", group_keys=False)
                   .tail(MAX_REV_USER)
        )
    print("After cleaning shape    :", reviews.shape)

    # 3) Merge on parent_asin
    merged = reviews.merge(meta, on="parent_asin", how="inner")
    print("After merge shape       :", merged.shape)

    # 4) Write as parquet shards
    write_shards(merged, OUT_DIR, PREFIX, SHARD_SIZE)
    print(f"Saved {merged.shape[0]:,} rows -> {OUT_DIR} as '{PREFIX}_NNNNN.parquet'")


if __name__ == "__main__":
    main()

