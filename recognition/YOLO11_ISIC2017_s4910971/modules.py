from pathlib import Path


def load_model(weights_path):
    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到本地模型权重：{weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path), task="detect")
    if model.task != "detect":
        raise ValueError("本脚本需要目标检测权重，而不是分类或分割权重。")
    return model
