"""Turns a flat, filled sheet into something that looks like a phone photo.

Every geometric step is tracked in one homography that maps template pixels to
photo pixels, so field boxes stay exact ground truth after the distortion.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BACKGROUNDS = (
    (0.55, 0.38, 0.24),  # wooden desk
    (0.70, 0.52, 0.34),  # light wood
    (0.45, 0.46, 0.48),  # grey table
    (0.18, 0.18, 0.20),  # dark desk
    (0.86, 0.86, 0.84),  # white table
)


@dataclass(frozen=True)
class PhotoOptions:
    long_side: int = 1600
    degrade: bool = True


def composite_ink(page: np.ndarray, mask: np.ndarray, ink: tuple[float, float, float], x: int, y: int) -> None:
    """Multiplies ink into `page` (float RGB in [0, 1]) at (x, y), clipping at the edges."""

    h, w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    alpha = mask[y0 - y : y1 - y, x0 - x : x1 - x, None]
    region = page[y0:y1, x0:x1]
    region *= 1 - alpha * (1 - np.asarray(ink, dtype=np.float32))


def _background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(BACKGROUNDS[rng.integers(len(BACKGROUNDS))], dtype=np.float32)
    base = np.clip(base + rng.normal(0, 0.04, 3), 0, 1)
    coarse = cv2.resize(rng.normal(0, 1, (8, 6)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), rng.uniform(0.8, 3))
    if rng.random() < 0.5:  # wood-like streaks
        streaks = cv2.resize(rng.normal(0, 1, (1, 64)).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        grain = grain + 1.5 * (streaks if rng.random() < 0.5 else cv2.resize(streaks.T, (width, height)))
    texture = 1 + 0.06 * coarse + 0.03 * grain
    return np.clip(base[None, None, :] * texture[..., None], 0, 1)


def _illumination(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    direction = (np.cos(angle) * (xs / width - 0.5) + np.sin(angle) * (ys / height - 0.5))
    light = 1 + rng.uniform(0, 0.35) * direction
    if rng.random() < 0.5:
        cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.2, 0.8) * height
        distance = np.hypot(xs - cx, ys - cy) / np.hypot(width, height)
        light *= 1 - rng.uniform(0.05, 0.3) * distance
    if rng.random() < 0.35:  # hand or phone shadow entering from one edge
        shadow = np.zeros((height, width), np.float32)
        edge = rng.integers(4)
        depth = rng.uniform(0.15, 0.45)
        polygon = {
            0: [(0, 0), (width, 0), (width * rng.uniform(0.3, 1), height * depth), (0, height * depth * rng.uniform(0.5, 1.5))],
            1: [(0, height), (width, height), (width, height * (1 - depth)), (0, height * (1 - depth * rng.uniform(0.5, 1.5)))],
            2: [(0, 0), (0, height), (width * depth, height * rng.uniform(0.3, 1)), (width * depth * rng.uniform(0.5, 1.5), 0)],
            3: [(width, 0), (width, height), (width * (1 - depth), height), (width * (1 - depth * rng.uniform(0.5, 1.5)), 0)],
        }[int(edge)]
        cv2.fillPoly(shadow, [np.array(polygon, np.int32)], 1.0)
        shadow = cv2.GaussianBlur(shadow, (0, 0), rng.uniform(10, 40))
        light *= 1 - rng.uniform(0.15, 0.45) * shadow
    return light


def photograph(page: np.ndarray, rng: np.random.Generator, options: PhotoOptions) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (photo as uint8 RGB after a JPEG round trip, 3x3 homography, parameters)."""

    page_h, page_w = page.shape[:2]
    out_h = options.long_side
    out_w = int(round(out_h * 3 / 4))
    params: dict = {"width": out_w, "height": out_h, "degraded": options.degrade}

    if not options.degrade:
        scale = min(out_w / page_w, out_h / page_h)
        small = cv2.resize(page, (int(page_w * scale), int(page_h * scale)), interpolation=cv2.INTER_AREA)
        offset_x, offset_y = (out_w - small.shape[1]) // 2, (out_h - small.shape[0]) // 2
        canvas = np.ones((out_h, out_w, 3), np.float32)
        canvas[offset_y : offset_y + small.shape[0], offset_x : offset_x + small.shape[1]] = small
        homography = np.array(
            [
                [small.shape[1] / page_w, 0, offset_x],
                [0, small.shape[0] / page_h, offset_y],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        photo = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        params["jpeg_quality"] = None
        return photo, homography, params

    for _ in range(50):
        fill = rng.uniform(0.72, 0.95)
        scale = min(fill * out_h / page_h, 0.97 * out_w / page_w)
        w, h = page_w * scale, page_h * scale
        center = np.array([out_w / 2, out_h / 2]) + rng.uniform(-1, 1, 2) * [(out_w - w) / 3, (out_h - h) / 3]
        angle = np.deg2rad(rng.normal(0, 3.5))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        tilt = rng.uniform(0, 0.06) * np.hypot(w, h)
        destination = corners @ rotation.T + center + rng.uniform(-1, 1, (4, 2)) * tilt
        if (destination >= 2).all() and (destination[:, 0] <= out_w - 2).all() and (destination[:, 1] <= out_h - 2).all():
            break
    else:  # pragma: no cover - the loop practically always finds a placement
        raise RuntimeError("could not place the page inside the photo")

    # Downscale first so the perspective warp does not alias thin strokes.
    small = cv2.resize(page, (int(page_w * scale), int(page_h * scale)), interpolation=cv2.INTER_AREA)
    source = np.float32([[0, 0], [small.shape[1], 0], [small.shape[1], small.shape[0]], [0, small.shape[0]]])
    warp = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    homography = warp @ np.diag([small.shape[1] / page_w, small.shape[0] / page_h, 1.0])

    warped = cv2.warpPerspective(small, warp, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    coverage = cv2.warpPerspective(np.ones(small.shape[:2], np.float32), warp, (out_w, out_h), flags=cv2.INTER_LINEAR)
    background = _background(out_w, out_h, rng)
    drop = cv2.GaussianBlur(np.roll(coverage, (int(rng.integers(3, 12)), int(rng.integers(3, 12))), (0, 1)), (0, 0), 8)
    background *= 1 - 0.35 * drop[..., None]
    photo = background * (1 - coverage[..., None]) + warped * coverage[..., None]

    paper_tint = np.array([1.0, rng.uniform(0.97, 1.0), rng.uniform(0.9, 1.0)], dtype=np.float32)
    photo = photo * (paper_tint * coverage[..., None] + (1 - coverage[..., None]))
    photo *= _illumination(out_w, out_h, rng)[..., None]
    # White balance: a warm/cool shift plus a slight green/magenta tint, as
    # phone cameras drift; independent per-channel gains give unreal colours.
    temperature, tint = rng.normal(0, 0.03), rng.normal(0, 0.008)
    photo *= np.array([1 + temperature, 1 + tint, 1 - temperature], dtype=np.float32)
    photo *= rng.uniform(0.85, 1.1)  # exposure

    blur = float(rng.uniform(0, 1.2)) if rng.random() < 0.7 else 0.0
    if blur > 0.1:
        photo = cv2.GaussianBlur(photo, (0, 0), blur)
    if rng.random() < 0.15:
        length = int(rng.integers(3, 9))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1 / length
        rotation_matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
        photo = cv2.filter2D(photo, -1, kernel / max(kernel.sum(), 1e-6))
        params["motion_blur"] = length
    noise = float(rng.uniform(0.004, 0.03))
    photo = photo + rng.normal(0, noise, photo.shape).astype(np.float32)

    quality = int(rng.integers(55, 96))
    photo = (np.clip(photo, 0, 1) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover
        raise RuntimeError("JPEG encoding failed")
    photo = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    params.update(
        {
            "page_fill": round(float(fill), 3),
            "rotation_deg": round(float(np.rad2deg(angle)), 2),
            "tilt_px": round(float(tilt), 1),
            "blur_sigma": round(blur, 2),
            "noise_sigma": round(noise, 4),
            "jpeg_quality": quality,
        }
    )
    return photo, homography, params


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def rectify(photo: np.ndarray, quad: np.ndarray, height: int, rng: np.random.Generator | None, jitter: float) -> np.ndarray:
    """Crops a field quad back to an upright rectangle, optionally with corner jitter
    that imitates an imperfect cell detector."""

    top_w = np.linalg.norm(quad[1] - quad[0])
    side_h = np.linalg.norm(quad[3] - quad[0])
    width = max(8, int(round(height * top_w / max(side_h, 1))))
    if rng is not None and jitter > 0:
        quad = quad + rng.normal(0, jitter, quad.shape) * [top_w, side_h]
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(photo, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
