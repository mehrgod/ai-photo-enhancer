"""
Image processing module.
Implements lightweight local enhancement operations for portraits.
"""

import cv2
import numpy as np
from PIL import Image

# Friendly names for each operation
OPERATION_NAMES = {
    "brightness_adjustment": "Brightness adjustment",
    "contrast_adjustment": "Contrast improvement",
    "face_sharpen": "Face-area sharpening",
    "noise_reduction": "Noise reduction",
    "auto_adjust": "Auto enhancement",
}


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV format (BGR)."""
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV format (BGR) to PIL Image (RGB)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def adjust_brightness(image: Image.Image, factor: float = 1.15) -> Image.Image:
    """
    Adjust brightness by scaling the V channel in HSV color space.
    factor > 1.0 makes the image brighter.
    """
    bgr = _pil_to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return _bgr_to_pil(adjusted)


def adjust_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Adjust contrast by scaling pixel values.
    factor > 1.0 increases contrast.
    """
    bgr = _pil_to_bgr(image)
    adjusted = cv2.convertScaleAbs(bgr, alpha=factor, beta=0)
    return _bgr_to_pil(adjusted)


def sharpen_image(image: Image.Image) -> Image.Image:
    """
    Sharpen the image using a simple unsharp mask kernel.
    Good for portrait faces.
    """
    bgr = _pil_to_bgr(image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")
    sharpened = cv2.filter2D(bgr, -1, kernel)
    return _bgr_to_pil(sharpened)


def reduce_noise(image: Image.Image) -> Image.Image:
    """
    Reduce noise using OpenCV's fastNlMeansDenoisingColored.
    Lightweight and effective on portraits.
    """
    bgr = _pil_to_bgr(image)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    return _bgr_to_pil(denoised)


def auto_adjust(image: Image.Image) -> Image.Image:
    """
    Apply a basic auto-enhancement: brightness + contrast.
    Good starting point for most photos.
    """
    result = adjust_brightness(image, factor=1.1)
    result = adjust_contrast(result, factor=1.15)
    return result


def apply_pipeline(image: Image.Image, pipeline: list) -> Image.Image:
    """
    Apply a sequence of enhancement operations to an image.
    pipeline: list of operation names like ['brightness_adjustment', 'face_sharpen']
    """
    operations = {
        "brightness_adjustment": adjust_brightness,
        "contrast_adjustment": adjust_contrast,
        "face_sharpen": sharpen_image,
        "noise_reduction": reduce_noise,
        "auto_adjust": auto_adjust,
    }

    current = image
    for step in pipeline:
        func = operations.get(step)
        if func is None:
            # Skip unknown operations gracefully
            continue
        current = func(current)
    return current
