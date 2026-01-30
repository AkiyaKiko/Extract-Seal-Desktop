# seal_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2


@dataclass
class Circle:
    x: float
    y: float
    radius: float


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip()
    if not (hex_color.startswith("#") and len(hex_color) == 7):
        raise ValueError("颜色格式应为 #RRGGBB，例如 #ff0000")
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return r, g, b


def rgb_to_hsv_opencv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    # OpenCV HSV: H[0..179], S[0..255], V[0..255]
    px = np.uint8([[[b, g, r]]])  # OpenCV expects BGR order for cvtColor
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def extract_stamp_to_bgra(img_bgr, color_hex="#ff0000", hue_range=10, s_min=50, v_min=50):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    r, g, b = hex_to_rgb(color_hex)
    h, _, _ = rgb_to_hsv_opencv(r, g, b)

    low_h = (h - hue_range + 180) % 180
    high_h = (h + hue_range) % 180

    if low_h > high_h:
        mask1 = cv2.inRange(hsv, (0, s_min, v_min), (high_h, 255, 255))
        mask2 = cv2.inRange(hsv, (low_h, s_min, v_min), (179, 255, 255))
        mask = cv2.add(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, (low_h, s_min, v_min), (high_h, 255, 255))

    out = np.zeros((img_bgr.shape[0], img_bgr.shape[1], 4), dtype=np.uint8)
    idx = mask > 0
    out[idx, 0] = b
    out[idx, 1] = g
    out[idx, 2] = r
    out[idx, 3] = 255

    return out, mask



def detect_circles(
    gray: np.ndarray,
    max_count: int = 6,
    param1: int = 200,
    param2: int = 50,
) -> List[Circle]:
    """
    复刻 JS detectCircles：
    - minRadius = min(h,w)*0.03
    - maxRadius = min(h,w)*0.5
    - minDist = rows/6
    - param1=200 param2=50
    """
    h, w = gray.shape[:2]
    min_radius = int(min(h, w) * 0.03)
    max_radius = int(min(h, w) * 0.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=h / 6.0,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    result: List[Circle] = []
    if circles is not None and circles.size > 0:
        circles = np.squeeze(circles, axis=0)
        for (x, y, r) in circles:
            result.append(Circle(float(x), float(y), float(r)))

    result.sort(key=lambda c: c.radius, reverse=True)
    return result[:max_count]


def crop_circle_bgra(
    stamp_bgra: np.ndarray,
    circle: Circle,
    scale_factor: float = 1.2,
) -> np.ndarray:
    """
    复刻 JS cropCircle：
    - 半径放大 scale_factor
    - ROI 截取
    - 圆形mask
    - alpha = alpha & mask，圆外透明
    返回：裁剪后的 BGRA
    """
    H, W = stamp_bgra.shape[:2]
    new_r = circle.radius * scale_factor
    size = int(round(new_r * 2))

    left = int(round(circle.x - new_r))
    top = int(round(circle.y - new_r))

    # 对齐 JS：把 ROI 起点限制在 [0, W-size] / [0, H-size]
    left = max(0, min(left, max(0, W - size)))
    top = max(0, min(top, max(0, H - size)))

    # ROI 实际宽高还要再 clamp 一次
    size_w = min(size, W - left)
    size_h = min(size, H - top)

    roi = stamp_bgra[top:top + size_h, left:left + size_w].copy()

    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    rad = int(min(roi.shape[0], roi.shape[1]) / 2)
    cv2.circle(mask, center, rad, 255, thickness=-1)

    alpha = roi[:, :, 3]
    roi[:, :, 3] = cv2.bitwise_and(alpha, mask)

    # 可选：把圆外 RGB 置零更干净
    inv = cv2.bitwise_not(mask)
    roi[inv > 0, 0:3] = 0

    return roi


def extract_seals(
    img_bgr: np.ndarray,
    color_hex: str = "#ff0000",
    hue_range: int = 10,
    s_min: int = 50,
    v_min: int = 50,
    blur_ksize: int = 5,
    blur_sigma: float = 2.0,
    scale_factor: float = 1.2,
    max_count: int = 6,
    hough_param1: int = 200,
    hough_param2: int = 50,
) -> Dict[str, object]:
    """
    一次性跑完整 pipeline：
    颜色分割(BGRA+Alpha) -> 灰度+高斯 -> 霍夫圆 -> 裁剪
    返回 dict，方便 UI 使用
    """
    stamp_bgra, mask = extract_stamp_to_bgra(
        img_bgr,
        color_hex=color_hex,
        hue_range=hue_range,
        s_min=s_min,
        v_min=v_min,
    )

    gray = cv2.cvtColor(stamp_bgra, cv2.COLOR_BGRA2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    gray = cv2.GaussianBlur(gray, (k, k), blur_sigma, blur_sigma)

    circles = detect_circles(
        gray,
        max_count=max_count,
        param1=hough_param1,
        param2=hough_param2,
    )

    crops = [crop_circle_bgra(stamp_bgra, c, scale_factor=scale_factor) for c in circles]

    return {
        "stamp_bgra": stamp_bgra,  # 纯色+透明背景图
        "mask": mask,              # 8-bit mask
        "gray": gray,              # 用于圆检测的灰度图
        "circles": circles,        # List[Circle]
        "crops": crops,            # List[np.ndarray BGRA]
    }
