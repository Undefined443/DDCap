#!/usr/bin/env python3
"""评估 ImageNet Caption 生成结果（参考 `utils/eval_coco_results.py`）。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTIONEVAL_ROOT = REPO_ROOT / "captioneval"
for candidate in (CAPTIONEVAL_ROOT, CAPTIONEVAL_ROOT / "coco_caption"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pycocoevalcap.eval import COCOEvalCap  # type: ignore  # noqa: E402
from pycocotools.coco import COCO  # type: ignore  # noqa: E402


def sanitize_caption(text: str) -> str:
    return text.split("<|endoftext|>", 1)[0].strip()


def resolve_prediction_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = path.with_suffix(".json")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"找不到预测文件：{raw_path}")


def load_predictions(pred_file: Path) -> OrderedDict[str, str]:
    with pred_file.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)

    predictions: "OrderedDict[str, str]" = OrderedDict()
    for item in entries:
        image_id = item.get("image_id")
        caption = item.get("caption") or item.get("result")
        if not image_id or caption is None:
            raise ValueError(f"无法识别的预测条目：{item}")
        if image_id in predictions:
            continue
        predictions[image_id] = sanitize_caption(caption)
    return predictions


def iter_parquet_rows(parquet_paths: Iterable[Path]) -> Iterable[Tuple[str, str]]:
    for path in parquet_paths:
        table = pq.read_table(path, columns=["image_id", "caption_enriched"])
        ids = table["image_id"].to_pylist()
        captions = table["caption_enriched"].to_pylist()
        for image_id, caption in zip(ids, captions):
            yield image_id, caption


def load_ground_truth(
    image_ids: Iterable[str], data_dir: Path, split: str
) -> Dict[str, str]:
    target = set(image_ids)
    found: Dict[str, str] = {}
    parquet_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"未在 {data_dir} 中找到以 {split}- 开头的 parquet 文件"
        )

    remaining = target.copy()
    for image_id, caption in iter_parquet_rows(parquet_paths):
        if image_id in remaining:
            if caption:
                found[image_id] = caption.strip()
            remaining.discard(image_id)
            if not remaining:
                break
    return found


def build_coco_inputs(
    image_ids: List[str],
    predictions: Dict[str, str],
    references: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    images: List[Dict] = []
    annotations: List[Dict] = []
    results: List[Dict] = []

    for idx, image_id in enumerate(image_ids):
        images.append({"id": idx, "file_name": image_id})
        annotations.append({"image_id": idx, "id": idx, "caption": references[image_id]})
        results.append({"image_id": idx, "caption": predictions[image_id]})
    return images, annotations, results


def evaluate_metrics(
    images: List[Dict],
    annotations: List[Dict],
    results: List[Dict],
) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        label_file = tmpdir_path / "labels.json"
        res_file = tmpdir_path / "results.json"

        label_content = {
            "info": {"description": "ImageNet Caption Evaluation"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
        }
        with label_file.open("w", encoding="utf-8") as fh:
            json.dump(label_content, fh)
        with res_file.open("w", encoding="utf-8") as fh:
            json.dump(results, fh)

        coco = COCO(str(label_file))
        coco_res = coco.loadRes(str(res_file))
        evaluator = COCOEvalCap(coco, coco_res, "corpus")
        evaluator.params["image_id"] = coco_res.getImgIds()
        evaluator.evaluate()
        return {metric: value * 100 for metric, value in evaluator.eval.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 ImageNet caption 生成结果")
    parser.add_argument("--predictions", required=True, help="预测结果 JSON")
    parser.add_argument(
        "--imagenet-dir",
        default="IMAGENET/data",
        help="ImageNet caption 数据 parquet 目录",
    )
    parser.add_argument(
        "--split", default="validation", help="parquet 文件前缀（如 validation/train）"
    )
    parser.add_argument(
        "--output",
        default="imagenet_evaluation_results.json",
        help="保存指标的 JSON 路径",
    )
    args = parser.parse_args()

    pred_path = resolve_prediction_path(args.predictions)
    print(f"加载预测结果：{pred_path}")
    predictions = load_predictions(pred_path)
    print(f"共加载 {len(predictions)} 条预测结果")

    data_dir = Path(args.imagenet_dir)
    print(f"\n加载 ImageNet {args.split} split ground truth...")
    references = load_ground_truth(predictions.keys(), data_dir, args.split)

    missing_refs = sorted(set(predictions) - set(references))
    if missing_refs:
        print(f"[warn] {len(missing_refs)} 条预测缺少 ground truth，将被跳过。")
        for sample in missing_refs[:5]:
            print(f"  - {sample}")

    effective_ids = [img_id for img_id in predictions if img_id in references]
    if not effective_ids:
        raise RuntimeError("无有效样本可评估。请检查 split 或数据路径。")

    images, annotations, results = build_coco_inputs(effective_ids, predictions, references)
    metrics = evaluate_metrics(images, annotations, results)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("评估结果：")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:6.2f}")
    print("=" * 60)
    print(f"\n详细结果已保存至：{output_path}")


if __name__ == "__main__":
    main()
