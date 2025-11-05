## COCO

转换COCO生成结果：

```sh
uv run utils/convert_coco_results.py --input_json <input.json> --output_json <prediction.json>
```

评估COCO生成结果:

```sh
uv run utils/eval_coco_results.py --predictions <prediction.json>
```

## ImageNet

转换ImageNet生成结果：

```sh
uv run utils/convert_imagenet_results.py \
    --epoch-log results/add-imagenet/epoch-500.txt \
    --imagenet-dir IMAGENET/data \
    --split validation \
    --output results/add-imagenet/epoch-500.json \
    --missing-output results/add-imagenet/epoch-500-missing.json
```

评估ImageNet生成结果:

```sh
uv run utils/eval_imagenet_results.py \
    --predictions results/add-imagenet/epoch-500.json \
    --imagenet-dir IMAGENET/data \
    --split validation \
    --output results/add-imagenet/eval-500.json
```