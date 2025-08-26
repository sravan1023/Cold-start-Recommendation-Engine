"""
clean_reviews_dir.py — Parquet cleaner with per-shard progress bars + checkpoint
===============================================================================

Input : data/pre_cleaned/*.parquet
Output: data/preprocessed/<shard>_{clean|dropped}.parquet (snappy)
Resume : data/preprocessed/_checkpoint.txt   ← one shard name per line
"""

from __future__ import annotations
import pandas as pd, numpy as np, ftfy, html, re, sys
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import emoji

# ── folders ──────────────────────────────────────────────────────────
IN_DIR  = Path("data/pre_cleaned")
OUT_DIR = Path("data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE = OUT_DIR / "_checkpoint.txt"

# ── regex & small helpers ────────────────────────────────────────────
VIDEO_RE = re.compile(r"\[\[VIDEOID:[^\]]+\]\]", re.I)
EXTRA    = {"\u200d", "\u20e3", "\ufe0f"}                  # helper glyphs

def strip_emoji(t: str) -> str:
    t = emoji.replace_emoji(t or "", replace="")
    return "".join(ch for ch in t if ch not in EXTRA)

def fix_mojibake(t: str) -> str:
    try:
        return t.encode("latin1", "ignore").decode("utf-8", "ignore")
    except Exception:
        return t

def norm_ws(t: str) -> str:
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"\s{2,}", " ", t)
    return re.sub(r"\s+([.,!?;:])", r"\1", t).strip()

def clean_text(raw):
    if pd.isna(raw):
        return raw
    t = BeautifulSoup(html.unescape(str(raw)), "lxml").get_text(" ", strip=True)
    t = VIDEO_RE.sub("", t)
    t = ftfy.fix_text(t)
    t = fix_mojibake(t)
    t = strip_emoji(t)
    return norm_ws(t)

def to_datetime(series: pd.Series) -> pd.Series:
    s  = pd.to_numeric(series, errors="coerce")
    ms = s[s > 1e11];  sc = s[s <= 1e11]
    out = pd.Series(index=series.index, dtype="datetime64[ns]")
    out.loc[ms.index] = pd.to_datetime(ms, unit="ms", errors="coerce", utc=True)
    out.loc[sc.index] = pd.to_datetime(sc, unit="s",  errors="coerce", utc=True)
    return out

# ── core cleaner for one DataFrame ───────────────────────────────────
def clean_df(df: pd.DataFrame, shard_name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop_duplicates().reset_index(drop=True)

    # text columns with per-column tqdm bars
    for col in ["text", "title", "description"]:
        df[col] = [
            clean_text(x)
            for x in tqdm(
                df[col],
                desc=f"{shard_name}:{col}",
                leave=False,
                unit="row",
            )
        ]

    df["timestamp"] = to_datetime(df["timestamp"])
    df["rating"]    = pd.to_numeric(df["rating"], errors="coerce")

    vp = df["verified_purchase"].astype(str).str.lower().str.strip()
    df["verified_purchase"] = np.where(vp == "true", True,
                               np.where(vp == "false", False, np.nan))

    # drop-row mask
    bad = (
        df["timestamp"].isna()
        | df["rating"].isna()
        | df["verified_purchase"].isna()
    )

    vp_false = df["verified_purchase"] == False
    short    = df["text"].str.len().fillna(0) < 15
    bad |= (vp_false & short)

    return df, bad

# ── checkpoint helpers ───────────────────────────────────────────────
def load_ckpt() -> set[str]:
    if CKPT_FILE.exists():
        with CKPT_FILE.open() as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def append_ckpt(name: str) -> None:
    with CKPT_FILE.open("a") as f:
        f.write(name + "\n")

# ── main loop ────────────────────────────────────────────────────────
def main() -> None:
    shards = sorted(IN_DIR.glob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet files in {IN_DIR.resolve()}")

    done = load_ckpt()
    todo = [p for p in shards if p.stem not in done]

    if not todo:
        print("All shards already processed.")
        return

    print(f"{len(done)} shard(s) done → {len(todo)} remaining.")

    for p in tqdm(todo, desc="Shards", unit="shard"):
        name   = p.stem
        df_raw = pd.read_parquet(p)

        df, mask = clean_df(df_raw, name)
        df[~mask].to_parquet(OUT_DIR / f"{name}_clean.parquet",
                             index=False, compression="snappy")
        df[mask].to_parquet(OUT_DIR / f"{name}_dropped.parquet",
                            index=False, compression="snappy")

        append_ckpt(name)   # mark shard complete in checkpoint

    print("\nDone — checkpoint updated.")

if __name__ == "__main__":
    main()



