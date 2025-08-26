# ── src/offline/build_popularity.py ───────────────────────────────────
"""
Build two popularity tables from cleaned shards (data_*.parquet):

• top_overall.parquet  – most-rated books across all history
• top_recent.parquet   – most-rated books in the last RECENT_DAYS days
"""

from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm

# -------- config ------------------------------------------------------
SHARDS       = "data/preprocessed/data_*.parquet"
OUT_DIR      = Path("data/popularity")
TOP_K        = 1_000          # rows to keep
RECENT_DAYS  = 90
# ----------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_counter, recent_counter = {}, {}
    recent_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=RECENT_DAYS)  # tz-aware already

    # 1) iterate shards
    for fp in tqdm(sorted(glob.glob(SHARDS)), desc="Counting ratings", unit="shard", colour="cyan"):
        df = pd.read_parquet(fp, columns=["parent_asin", "timestamp"])

        # overall
        vc = df["parent_asin"].value_counts()
        for asin, cnt in vc.items():
            overall_counter[asin] = overall_counter.get(asin, 0) + cnt

        # recent
        recent = df[df["timestamp"] >= recent_cutoff]
        vc_r   = recent["parent_asin"].value_counts()
        for asin, cnt in vc_r.items():
            recent_counter[asin] = recent_counter.get(asin, 0) + cnt

    # 2) top-K DataFrames
    overall = (pd.Series(overall_counter, name="n_ratings")
                 .sort_values(ascending=False)
                 .head(TOP_K).rename_axis("parent_asin").reset_index())

    recent  = (pd.Series(recent_counter, name="n_ratings_recent")
                 .sort_values(ascending=False)
                 .head(TOP_K).rename_axis("parent_asin").reset_index())

    # 3) join title / description once (fast)
    first_meta = (pd.read_parquet(sorted(glob.glob(SHARDS))[0],
                                  columns=["parent_asin", "title", "description"])
                    .drop_duplicates("parent_asin"))
    overall = overall.merge(first_meta, on="parent_asin", how="left")
    recent  = recent.merge(first_meta,  on="parent_asin", how="left")

    # 4) write
    overall.to_parquet(OUT_DIR / "top_overall.parquet", index=False)
    recent.to_parquet(OUT_DIR / "top_recent.parquet",  index=False)
    print("Popularity tables written to:", OUT_DIR)


if __name__ == "__main__":
    main()

