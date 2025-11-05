"""
通用COCO Caption评估脚本 - 可用于评估任何图像描述模型

使用示例：
python eval_other_model.py --predictions my_model_output.json
"""

import json
import argparse
from misc import evaluate_on_coco_caption


def load_predictions(pred_file):
    """
    加载其他模型的预测结果

    支持两种格式：
    1. 标准COCO结果格式：[{"image_id": 139, "caption": "..."}]
    2. 本项目格式：[{"image_id": "000000000139.jpg", "result": "..."}]
    """
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    # 转换为本项目格式
    results = []
    for pred in predictions:
        if 'caption' in pred:  # 标准COCO格式
            # 转换为项目格式
            img_id = pred['image_id']
            if isinstance(img_id, int):
                # 转换为文件名格式
                img_id = f"{img_id:012d}.jpg"
            results.append({
                'image_id': img_id,
                'result': pred['caption']
            })
        elif 'result' in pred:  # 本项目格式
            results.append(pred)
        else:
            raise ValueError(f"无法识别的预测格式：{pred}")

    return results


def main():
    parser = argparse.ArgumentParser(description='评估图像描述模型在COCO数据集上的表现')
    parser.add_argument('--predictions', required=True, help='模型预测结果文件（JSON格式）')
    parser.add_argument('--ground_truth', default='./MSCOCO_Caption/annotations/captions_val2014.json',
                        help='COCO ground truth文件路径')
    parser.add_argument('--output', default='evaluation_results.json', help='保存评估结果的路径')
    args = parser.parse_args()

    print(f"加载预测结果：{args.predictions}")
    results = load_predictions(args.predictions)
    print(f"共加载 {len(results)} 条预测结果")

    print(f"\n使用ground truth：{args.ground_truth}")
    print("开始评估...")

    # 调用评估函数
    metrics = evaluate_on_coco_caption(
        results=results,
        res_file=args.output.replace('.json', '_formatted.json'),
        label_file=args.ground_truth,
        outfile=args.output
    )

    print("\n" + "="*60)
    print("评估结果：")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:6.2f}")
    print("="*60)
    print(f"\n详细结果已保存至：{args.output}")


if __name__ == '__main__':
    main()
