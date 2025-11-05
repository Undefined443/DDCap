#!/usr/bin/env python3
"""
Convert ImageNet epoch logs to COCO-style prediction format.

Usage:
    python convert_imagenet_epoch_to_json.py \
        --epoch-log results/add-imagenet/epoch-500.txt \
        --imagenet-dir IMAGENET/data \
        --split validation \
        --output epoch_500_imagenet.json
"""

from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq

TRANSLATOR = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace."""
    return " ".join(text.lower().translate(TRANSLATOR).split())


def iter_caption_batches(
    parquet_paths: Iterable[Path],
    columns: Tuple[str, str] = ("image_id", "caption_enriched"),
) -> Iterable[Tuple[str, str]]:
    """Yield (caption, image_id) pairs from a collection of parquet files."""
    col_image, col_caption = columns
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=[col_image, col_caption])
        ids = table[col_image].to_pylist()
        caps = table[col_caption].to_pylist()
        for image_id, caption in zip(ids, caps):
            if caption:
                yield caption, image_id


def build_caption_index(
    data_dir: Path,
    split_prefix: str,
    columns: Tuple[str, str] = ("image_id", "caption_enriched"),
) -> Dict[str, List[Tuple[str, str]]]:
    """Build normalized caption → [(caption, image_id)] map for fast lookup."""
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    parquet_paths = sorted(data_dir.glob(f"{split_prefix}-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files like {split_prefix}-*.parquet found under {data_dir}"
        )

    for caption, image_id in iter_caption_batches(parquet_paths, columns=columns):
        index[normalize(caption)].append((caption, image_id))
    return index


def parse_epoch_log(epoch_log: Path) -> List[Tuple[str, str]]:
    """Extract (origin_caption, generated_caption) pairs from the epoch log."""
    entries: List[Tuple[str, str]] = []
    current_origin: str | None = None
    with epoch_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if "origin text:" in line:
                current_origin = line.split("origin text:", 1)[1].strip()
            elif "decode text:" in line and current_origin is not None:
                generated = line.split("decode text:", 1)[1].strip()
                entries.append((current_origin, generated))
                current_origin = None
    return entries


def match_image_id(
    origin_caption: str, index: Dict[str, List[Tuple[str, str]]]
) -> Tuple[str, str] | None:
    """Return (image_id, match_method) for a given origin caption."""
    norm = normalize(origin_caption)
    matches = index.get(norm)
    method = "normalized equality"

    if matches is None:
        for norm_caption, items in index.items():
            if norm and norm in norm_caption:
                matches = items
                method = "substring"
                break

    if not matches:
        return None

    unique = sorted({img_id for _, img_id in matches})
    if len(unique) > 1:
        duplicate_info = ", ".join(unique)
        print(
            f"[warn] Caption '{origin_caption}' matched multiple image_ids: {duplicate_info}"
        )

    return unique[0], method


def sanitize_caption(caption: str) -> str:
    return caption.split("<|endoftext|>", 1)[0].strip()


def convert(
    epoch_log: Path, data_dir: Path, split_prefix: str
) -> Tuple[List[Dict[str, str]], List[str]]:
    index = build_caption_index(data_dir, split_prefix)
    pairs = parse_epoch_log(epoch_log)

    predictions: List[Dict[str, str]] = []
    missing: List[str] = []
    for origin, generated in pairs:
        match = match_image_id(origin, index)
        if match is None:
            missing.append(origin)
            continue
        image_id, _ = match
        predictions.append(
            {
                "image_id": image_id,
                "caption": sanitize_caption(generated),
            }
        )
    return predictions, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epoch-log",
        type=Path,
        required=True,
        help="Path to the ImageNet epoch log containing origin/decode pairs.",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=Path("IMAGENET/data"),
        help="Directory with ImageNet parquet shards (train-XXXX / validation-XXXX).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Shard prefix to scan (e.g. validation, train).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; prints to stdout if omitted.",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help="Optional path to record origin captions that could not be matched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, missing = convert(args.epoch_log, args.imagenet_dir, args.split)

    if args.output:
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(predictions, fh, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

    if missing:
        print(f"[info] Skipped {len(missing)} entries that could not be matched.")
        if args.missing_output:
            with args.missing_output.open("w", encoding="utf-8") as fh:
                json.dump(missing, fh, ensure_ascii=False, indent=2)
        else:
            for caption in missing:
                print(f"  - {caption}")


if __name__ == "__main__":
    main()
