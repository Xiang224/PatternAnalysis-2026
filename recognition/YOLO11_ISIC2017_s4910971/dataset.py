import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data/ISIC2017"
CLASS_NAMES = ("melanoma", "nevus", "seborrheic_keratosis")
SPLITS = {"train": "Training", "val": "Validation", "test": "Test_v2"}


def mask_to_bbox(mask: Image.Image) -> Tuple[int, int, int, int]:
    if mask.mode not in ("1", "L"):
        raise ValueError(f"需要单通道二值掩码，实际图像模式为 {mask.mode}")

    colors = mask.getcolors(maxcolors=3)
    values = {value for _, value in colors} if colors is not None else set()
    if not values or not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("掩码必须只包含 0/1 或 0/255，不能使用原图或超像素图")

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("掩码没有前景像素，无法计算病灶框")
    return bbox


def bbox_to_yolo(
    bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽高必须大于 0")
    xmin, ymin, xmax, ymax = bbox
    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
        raise ValueError("框必须有正宽高，且不能超出图片边界")

    center_x = (xmin + xmax) / (2 * image_width)
    center_y = (ymin + ymax) / (2 * image_height)
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height
    return center_x, center_y, box_width, box_height


def read_classes(csv_path: Path):
    mapping = {(1.0, 0.0): 0, (0.0, 0.0): 1, (0.0, 1.0): 2}
    classes = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "melanoma", "seborrheic_keratosis"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"类别 CSV 缺少必要列：{csv_path}")
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in classes:
                raise ValueError(f"CSV 图片编号为空或重复：{image_id!r}")
            try:
                flags = (float(row["melanoma"]), float(row["seborrheic_keratosis"]))
                classes[image_id] = mapping[flags]
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{image_id} 的疾病类别无效，不能默认为其他类别") from None
    if not classes:
        raise ValueError(f"类别 CSV 没有样本：{csv_path}")
    return classes


def prepare_dataset(source: Path, output: Path):
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}。可用 --output 指定新目录")

    samples = {}
    seen_ids = set()
    for split, official_name in SPLITS.items():
        image_dir = source / "raw" / f"ISIC-2017_{official_name}_Data"
        mask_dir = source / "raw" / f"ISIC-2017_{official_name}_Part1_GroundTruth"
        csv_path = source / "metadata" / f"ISIC-2017_{official_name}_Part3_GroundTruth.csv"
        classes = read_classes(csv_path)
        image_ids = {path.stem for path in image_dir.glob("*.jpg")}
        mask_ids = {path.name.removesuffix("_segmentation.png")
                    for path in mask_dir.glob("*_segmentation.png")}
        if image_ids != set(classes) or mask_ids != set(classes):
            raise ValueError(f"{split} 的 JPG、掩码和 CSV 图片编号不完全匹配，停止转换")
        overlap = seen_ids & image_ids
        if overlap:
            raise ValueError(f"不同数据集合出现重复图片编号：{sorted(overlap)[:5]}")
        seen_ids.update(image_ids)

        samples[split] = []
        for image_id in sorted(image_ids):
            image_path = image_dir / f"{image_id}.jpg"
            with Image.open(mask_dir / f"{image_id}_segmentation.png") as mask:
                bbox = mask_to_bbox(mask)
                size = mask.size
            with Image.open(image_path) as original:
                if original.size != size:
                    raise ValueError(f"{image_id} 原图与掩码尺寸不一致")
            coordinates = bbox_to_yolo(bbox, size)
            class_id = classes[image_id]
            label = f"{class_id} " + " ".join(f"{v:.8f}" for v in coordinates) + "\n"
            samples[split].append((image_path, label, class_id))
        print(f"已检查 {split}：{len(samples[split])} 张", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    for split, records in samples.items():
        images_dir = output / "images" / split
        labels_dir = output / "labels" / split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        counts = [0] * len(CLASS_NAMES)
        for image_path, label, class_id in records:
            shutil.copy2(image_path, images_dir / image_path.name)
            with (labels_dir / f"{image_path.stem}.txt").open("x", encoding="utf-8") as handle:
                handle.write(label)
            counts[class_id] += 1
        print(f"已导出 {split}：{len(records)} 张；类别计数 {dict(zip(CLASS_NAMES, counts))}", flush=True)

    config = (
        f"path: {json.dumps(str(output), ensure_ascii=False)}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    )
    yaml_path = output / "data.yaml"
    with yaml_path.open("x", encoding="utf-8") as handle:
        handle.write(config)
    print(f"完成。YOLO 训练时将 data 参数指向：{yaml_path}")
    print("保留官方划分，未验证患者级独立性；未启动训练或计算测试集指标。")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="将 ISIC2017 转换为 YOLO 检测数据；--mask 仅预览单张掩码")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA, help="含 raw/metadata 的 ISIC2017 目录")
    parser.add_argument("--output", type=Path, help="输出目录，默认为 source/yolo；拒绝覆盖已有目录")
    parser.add_argument("--mask", type=Path, help="只预览指定二值掩码的框，不生成数据集")
    parser.add_argument("--image", type=Path, help="可选：对应原图路径，用于核对尺寸")
    args = parser.parse_args()
    if args.image is not None and args.mask is None:
        parser.error("--image 仅用于 --mask 单图预览模式")

    try:
        if args.mask is None:
            prepare_dataset(args.source, args.output if args.output is not None else args.source / "yolo")
            return
        with Image.open(args.mask.expanduser()) as mask:
            image_size = mask.size
            bbox = mask_to_bbox(mask)
        if args.image is not None:
            with Image.open(args.image.expanduser()) as original:
                if original.size != image_size:
                    raise ValueError(f"原图尺寸 {original.size} 与掩码尺寸 {image_size} 不一致")
        normalized_bbox = bbox_to_yolo(bbox, image_size)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"掩码：{args.mask.expanduser().resolve()}")
    print(f"图片尺寸 (宽, 高)：{image_size}")
    print(f"像素框 (xmin, ymin, xmax, ymax)：{bbox}（右/下边界不包含）")
    print("YOLO 框 (中心x, 中心y, 宽, 高)：" + " ".join(f"{v:.6f}" for v in normalized_bbox))
    print("这里只计算位置；类别需从 CSV 获取。未生成任何标签文件。")


if __name__ == "__main__":
    main()
