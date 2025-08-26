# ── src/data/preprocess.py ────────────────────────────────────────────
"""
Amazon Books  >  Parquet shards  (resumable, column-selected)

Quick run (uses defaults):
    python -m src.data.preprocess

Override with your own files:
    python -m src.data.preprocess  other/reviews.jsonl  other/meta.jsonl
"""

from __future__ import annotations
import argparse, gzip, json, orjson, time, typing as T
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ── default raw-file locations ───────────────────────────────────────
DEFAULT_REVIEWS = Path("data/raw/review_books.jsonl")
DEFAULT_META    = Path("data/raw/meta_books.jsonl")

# ── columns we keep ──────────────────────────────────────────────────
REV_COLS = {
    "user_id": None,
    "parent_asin": None,
    "timestamp": None,
    "rating": None,
    "text": "",
    "verified_purchase": False,
}
META_COLS = {
    "parent_asin": None,
    "title": None,
    "description": [],
    "price": None,
    "categories": [],
}

CHUNK_SIZE = 250_000


# ── helpers (stream, write, checkpoint) ──────────────────────────────
def _stream(path: Path) -> T.Iterator[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rb") as f:
        peek = f.peek(1) if hasattr(f, "peek") else f.read(1)
        f.seek(0)
        if peek.lstrip()[:1] == b"[":          # big JSON array
            for obj in json.load(f):
                yield obj
        else:                                  # JSONL
            for ln in f:
                if ln.strip():
                    yield orjson.loads(ln)


def _write(df: pd.DataFrame, out_dir: Path, prefix: str, idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{prefix}_{idx:05d}.parquet", index=False)


def _ckpt_load(out_dir: Path) -> dict:
    p = out_dir / "ckpt.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _ckpt_save(out_dir: Path, ck: dict):
    (out_dir / "ckpt.json").write_text(json.dumps(ck))


def _pass(in_path: Path, prefix: str, cols: dict, out_dir: Path, ck: dict):
    done_lines  = ck.get(prefix, {}).get("lines", 0)
    done_chunks = ck.get(prefix, {}).get("chunks", 0)
    buf = {k: [] for k in cols}

    bar = tqdm(desc=f"{prefix:8}", unit="ln", initial=done_lines, colour="cyan")
    for i, obj in enumerate(_stream(in_path)):
        if i < done_lines:
            continue

        for k, default in cols.items():
            val = obj.get(k, default)
            # Fix price issues
            if k == "price":
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = None

            # Flatten list fields
            if k in ("description", "categories") and isinstance(val, list):
                val = " ".join(val)
            buf[k].append(val)

        if len(buf[k]) >= CHUNK_SIZE:
            _write(pd.DataFrame(buf), out_dir, prefix, done_chunks)
            done_chunks += 1
            for lst in buf.values():
                lst.clear()
            ck[prefix] = {"lines": i + 1, "chunks": done_chunks}
            _ckpt_save(out_dir, ck)

        bar.update(1)

    if buf[next(iter(buf))]:
        _write(pd.DataFrame(buf), out_dir, prefix, done_chunks)
        done_chunks += 1
    ck[prefix] = {"lines": i + 1, "chunks": done_chunks}
    _ckpt_save(out_dir, ck)
    bar.close()


# ── main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser("preprocess raw Amazon-Books dumps")
    p.add_argument(
        "reviews", nargs="?", type=Path, default=DEFAULT_REVIEWS,
        help=f"reviews file (default: {DEFAULT_REVIEWS})"
    )
    p.add_argument(
        "meta", nargs="?", type=Path, default=DEFAULT_META,
        help=f"meta file    (default: {DEFAULT_META})"
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    args = p.parse_args(argv)

    ck = _ckpt_load(args.out_dir)
    t0 = time.time()

    print("Reviews pass")
    _pass(args.reviews, "reviews", REV_COLS, args.out_dir, ck)
    print("Meta pass")
    _pass(args.meta, "meta", META_COLS, args.out_dir, ck)

    print(f"finished in {time.time() - t0:,.1f}s")


if __name__ == "__main__":   # allows `python -m src.data.preprocess`
    main()

