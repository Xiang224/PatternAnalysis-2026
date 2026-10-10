import argparse
from pathlib import Path

import yaml

from dataset import CLASS_NAMES, REPO_ROOT
from modules import load_model


def check_data_config(data_path):
    with data_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("数据配置必须是 YAML 字典")
    expected = dict(enumerate(CLASS_NAMES))
    if config.get("nc") != 3 or config.get("names") != expected:
        raise ValueError(f"数据配置必须使用已导出标签的三类别及编号：{expected}")
    root = Path(config.get("path", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("data.yaml 的 path 必须是存在的数据根目录绝对路径")
    splits = {}
    for split in ("train", "val"):
        relative = config.get(split)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"配置缺少 {split} 图片目录")
        images = (root / relative).resolve()
        if not images.is_dir() or not any(images.glob("*.jpg")):
            raise ValueError(f"找不到 {split} 图片：{images}")
        labels = root / "labels" / split
        if not labels.is_dir() or not any(labels.glob("*.txt")):
            raise ValueError(f"找不到 {split} 标签：{labels}")
        splits[split] = images
    if splits["train"] == splits["val"]:
        raise ValueError("训练集与验证集不能使用同一图片目录")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="微调 YOLO11 病灶检测模型；默认训练 3 轮，--check 只检查不训练")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/ISIC2017/yolo/data.yaml")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=3, help="默认 3 轮只用于跑通流程")
    parser.add_argument("--batch", type=int, default=4, help="每批图片数量")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，必须是 32 的正整数倍")
    parser.add_argument("--device", default="cpu", help="cpu、Apple 芯片的 mps，或 CUDA 编号 0")
    parser.add_argument("--workers", type=int, default=0, help="数据加载进程数，默认 0 便于本地调试")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="isic2017_baseline", help="runs 下的实验目录名")
    parser.add_argument("--check", action="store_true", help="只检查，不训练")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch <= 0:
        parser.error("--epochs 和 --batch 必须大于 0")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz 必须是 32 的正整数倍")
    if args.workers < 0:
        parser.error("--workers 不能为负数")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name 必须是单个文件夹名称，不能是路径")
    args.data = args.data.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    if not args.weights.is_file():
        parser.error(f"找不到本地模型权重：{args.weights}")
    try:
        check_data_config(args.data)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    model = load_model(args.weights)
    print(f"初始权重：{args.weights}（{len(model.names)} 类）")
    print(f"目标类别：{dict(enumerate(CLASS_NAMES))}")
    print(f"数据配置：{args.data}")
    print(f"参数：epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, device={args.device}")
    if args.check:
        print("检查通过。尚未适配分类头或启动训练；未验证训练设备可用性。")
        return

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=True,
        freeze=0,
        optimizer="auto",
        amp=False,
        cache=False,
        val=True,
        save=True,
        project=str(REPO_ROOT / "runs"),
        name=args.name,
        exist_ok=False,
    )
    print(f"训练结果目录：{model.trainer.save_dir}")
    print(f"最佳验证权重：{model.trainer.best}")
    print(f"最后一轮权重：{model.trainer.last}")
    print("预测时用 --weights 指定上述最佳权重。未使用测试集评估。")


if __name__ == "__main__":
    main()
