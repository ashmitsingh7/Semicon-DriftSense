"""I/O utilities for images and JSON."""

import cv2
import json
import numpy as np


def load_image(path, as_float=True):
    """Load image as float32 [0,1] or uint8 [0,255]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if as_float:
        return img.astype(np.float32) / 255.0
    return img


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)