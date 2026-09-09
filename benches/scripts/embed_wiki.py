#!/usr/bin/env python3
"""Encode benches/wiki.json with a local sentence-transformers model (from `hf download`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki",
        type=Path,
        default=root / "wiki.json",
        help="NDJSON with url, title, body per line",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "hf_models" / "all-MiniLM-L6-v2",
        help="Local HF snapshot (download with: hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir benches/hf_models/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--out-bin",
        type=Path,
        default=root / "wiki_embedded.f32.bin",
        help="Row-major f32 little-endian bytes",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=root / "wiki_embedded.meta.json",
        help="JSON sidecar with dimension, count, model id",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(
            f"Model directory not found: {args.model}\n"
            "Run: hf download sentence-transformers/all-MiniLM-L6-v2 "
            f"--local-dir {args.model.parent / 'all-MiniLM-L6-v2'}"
        )

    from sentence_transformers import SentenceTransformer

    texts: list[str] = []
    with args.wiki.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            body = (obj.get("body") or "").strip()
            texts.append(f"{title}\n{body}")

    model = SentenceTransformer(str(args.model))
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    import numpy as np

    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"Expected 2D embedding matrix, got shape {arr.shape}")
    if arr.shape[0] != len(texts):
        raise SystemExit("Embedding row count does not match input text count")

    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.out_bin)

    meta = {
        "dimension": int(arr.shape[1]),
        "count": int(arr.shape[0]),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    args.out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_bin} ({arr.nbytes} bytes)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
