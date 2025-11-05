#!/usr/bin/env python3
"""
Convert epoch caption logs to COCO official prediction format.

Usage:
    python convert_epoch_to_json.py \
        --epoch-log results/add/epoch-2100.txt \
        --coco-caption MSCOCO_Caption/annotations/captions_val2014.json \
        --output epoch_2100_coco.json
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Dict, List, Tuple

TRANSLATOR = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for matching."""
    return " ".join(text.lower().translate(TRANSLATOR).split())


def build_caption_index(coco_path: Path) -> Dict[str, List[Tuple[str, int]]]:
    """Index COCO annotations by their normalized caption."""
    with coco_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    index: Dict[str, List[Tuple[str, int]]] = {}
    for ann in data["annotations"]:
        norm = normalize(ann["caption"])
        index.setdefault(norm, []).append((ann["caption"], ann["image_id"]))
    return index


def parse_epoch_log(epoch_path: Path) -> List[Tuple[str, str]]:
    """Extract (original, generated) caption pairs from the epoch log."""
    entries: List[Tuple[str, str]] = []
    pending_origin: str | None = None

    with epoch_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if "origin text:" in line:
                pending_origin = line.split("origin text:", 1)[1].strip()
            elif "decode text:" in line and pending_origin is not None:
                generated = line.split("decode text:", 1)[1].strip()
                entries.append((pending_origin, generated))
                pending_origin = None

    return entries


def match_image_id(origin: str, index: Dict[str, List[Tuple[str, int]]]) -> int:
    """Find the COCO image ID for a given origin caption."""
    norm_origin = normalize(origin)
    matches = index.get(norm_origin)

    if matches is None:
        for norm_caption, entries in index.items():
            if norm_origin and norm_origin in norm_caption:
                matches = entries
                break

    if not matches:
        raise KeyError(f"No COCO caption found for origin: {origin!r}")

    image_ids = sorted({image_id for _, image_id in matches})
    if not image_ids:
        raise KeyError(f"No image IDs associated with origin: {origin!r}")

    return image_ids[0]


def sanitize_caption(caption: str) -> str:
    """Trim special tokens to align with COCO evaluation expectations."""
    return caption.split('<|endoftext|>', 1)[0].strip()


def convert(epoch_log: Path, coco_caption: Path) -> List[dict]:
    """Return captions in COCO official `[{'image_id': int, 'caption': str}, ...]` format."""
    caption_index = build_caption_index(coco_caption)
    paired_captions = parse_epoch_log(epoch_log)

    coco_formatted = []
    for original, generated in paired_captions:
        image_id = match_image_id(original, caption_index)
        coco_formatted.append({
            "image_id": image_id,
            "caption": sanitize_caption(generated)
        })
    return coco_formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epoch-log",
        type=Path,
        default=Path("results/add/epoch-2100.txt"),
        help="Path to epoch log containing origin/decode text pairs.",
    )
    parser.add_argument(
        "--coco-caption",
        type=Path,
        default=Path("MSCOCO_Caption/annotations/captions_val2014.json"),
        help="Path to COCO captions JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON output; prints to stdout if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_predictions = convert(args.epoch_log, args.coco_caption)

    if args.output:
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(coco_predictions, fh, ensure_ascii=True, indent=2)
    else:
        print(json.dumps(coco_predictions, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
