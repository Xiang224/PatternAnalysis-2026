"""Pretrained model loading. Adapted from the official Ultralytics YOLO11 API.

Reference: https://docs.ultralytics.com/models/yolo11/
This is setup code, not a completed lesion-specific architecture implementation.
"""
from pathlib import Path

def load_pretrained_detector(weights_path):
    """Load an explicitly downloaded local detection checkpoint, without training."""
    from ultralytics import YOLO
    weights = Path(weights_path).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f'Download the official weights first: {weights}')
    model = YOLO(str(weights), task='detect')
    if model.task != 'detect':
        raise ValueError(f'Expected a detection checkpoint, got {model.task}')
    return model

