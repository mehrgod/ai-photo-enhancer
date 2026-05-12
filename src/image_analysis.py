"""
Image analysis module.
Extracts metrics: brightness, blur, noise from a portrait image.
"""

import cv2
import numpy as np
from PIL import Image


def analyze_image(image: Image.Image) -> dict:
    """
    Analyze an image and return metrics: brightness, blur, noise.
    Returns a dict with human-readable strings and raw numeric values.
    """
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Calculate metrics
    brightness_value = float(np.mean(gray))
    blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise_value = float(
        np.std(gray.astype("float32") - cv2.GaussianBlur(gray, (5, 5), 0))
    )

    return {
        "brightness": _describe_brightness(brightness_value),
        "blur": _describe_blur(blur_value),
        "noise": _describe_noise(noise_value),
        "raw": {
            "brightness_value": round(brightness_value, 1),
            "blur_value": round(blur_value, 1),
            "noise_value": round(noise_value, 1),
        },
    }


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV format (BGR)."""
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _describe_brightness(value: float) -> str:
    """Classify brightness as 'low', 'medium', or 'high'."""
    if value < 90:
        return "low"
    if value < 150:
        return "medium"
    return "high"


def _describe_blur(value: float) -> str:
    """Classify blur level using Laplacian variance."""
    if value < 100:
        return "high"
    if value < 300:
        return "medium"
    return "low"


def _describe_noise(value: float) -> str:
    """Classify noise level by comparing with Gaussian blur."""
    if value > 15:
        return "high"
    if value > 7:
        return "medium"
    return "low"
