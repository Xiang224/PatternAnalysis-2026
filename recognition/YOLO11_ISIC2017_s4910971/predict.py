import argparse
from pathlib import Path

from modules import load_model
from dataset import CLASS_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="使用微调权重预测病灶；--demo 使用原始通用模型")
    parser.add_argument("--demo", action="store_true", help="运行原来的通用图片验证，而非病灶预测")
    parser.add_argument("--source", type=Path,
                        help="本地图片文件或图片目录")
    parser.add_argument("--weights", type=Path,
                        help="本地检测模型权重")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="保留检测结果的最低置信度，范围 0 到 1")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理输入尺寸，要求为 32 的正整数倍")
    parser.add_argument("--device", default="cpu",
                        help="cpu、Apple 芯片的 mps，或 CUDA 设备编号 0")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0 到 1 之间")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz 必须是 32 的正整数倍")

    if args.source is None:
        args.source = (REPO_ROOT / "data/pretrained_demo" if args.demo else
                       REPO_ROOT / "data/ISIC2017/yolo/images/val/ISIC_0001769.jpg")
    if args.weights is None:
        args.weights = (REPO_ROOT / "weights/yolo11n.pt" if args.demo else
                        REPO_ROOT / "runs/isic2017_baseline/weights/best.pt")
    args.weights = args.weights.expanduser().resolve()
    if not args.weights.is_file():
        parser.error(f"找不到权重：{args.weights}。请先训练，或用 --weights 指定已有权重；"
                     "原来的通用模型验证请加 --demo。")
    args.source = args.source.expanduser().resolve()
    if args.source.is_file():
        if args.source.suffix.lower() not in IMAGE_SUFFIXES:
            parser.error("--source 文件必须是 JPG、JPEG、PNG、BMP 或 WEBP 图片")
    elif args.source.is_dir():
        if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                   for p in args.source.iterdir()):
            parser.error(f"目录中没有支持的图片：{args.source}")
    else:
        parser.error(f"找不到输入图片或目录：{args.source}")
    return args


def main(argv=None):
    args = parse_args(argv)
    model = load_model(args.weights)
    if not args.demo and model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError("病灶预测需要类别及编号匹配的三类微调权重，不能使用原始 80 类权重。"
                         "通用模型验证请加 --demo。")
    print(f"加载模型：{args.weights.resolve()}（{len(model.names)} 类）")
    print(f"设备：{args.device}；输入尺寸：{args.imgsz}；置信度阈值：{args.conf}")

    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        save=True,
        project=str(REPO_ROOT / "runs"),
        name="pretrained_demo" if args.demo else "lesion_prediction",
        exist_ok=False,
        verbose=False,
    )

    image_count = 0
    for result in results:
        image_count += 1
        print(f"\n图片：{Path(result.path).name}")
        if result.boxes is None or len(result.boxes) == 0:
            print("  当前阈值下没有检测结果。")
        else:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coordinates = [round(value, 1) for value in box.xyxy[0].tolist()]
                print(f"  {result.names[class_id]} | 置信度 {confidence:.3f}"
                      f" | xyxy={coordinates}")
        print(f"  画框结果目录：{result.save_dir}")

    print(f"\n完成 {image_count} 张图片的推理。未训练模型，未计算检测准确率。")


if __name__ == "__main__":
    main()
