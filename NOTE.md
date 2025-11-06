## 训练

```sh
torchrun --nproc-per-node auto train.py --out_dir results --tag caption_diff_vitb16
```

## COCO

转换COCO生成结果：

```sh
input="/path/to/epoch-log"
output="/path/to/prediction"
uv run utils/convert_coco_results.py --epoch-log $input --output $output
```

评估COCO生成结果:

```sh
export PYTHONPATH="$PWD"
input="/path/to/prediction"
output="/path/to/result"
uv run utils/eval_coco_results.py --predictions $input --output $output
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
