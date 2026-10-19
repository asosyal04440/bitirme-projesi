"""Synthetic handwriting: renders text as an ink alpha mask.

Two glyph sources are supported:
- handwriting-style system fonts (always available, no download), with
  per-character rotation, scale, baseline and spacing jitter, slant, elastic
  distortion and stroke-width variation so the same font never repeats exactly;
- optionally, real handwritten digits from MNIST, EMNIST or USPS style
  datasets, used for digits only (names and punctuation still come from fonts).
"""

from __future__ import annotations

import gzip
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Handwriting-like fonts. They are read from the system and never redistributed.
CANDIDATE_FONTS = (
    "Inkfree.ttf",
    "segoepr.ttf",
    "segoeprb.ttf",
    "segoesc.ttf",
    "segoescb.ttf",
    "comic.ttf",
    "comici.ttf",
    "mvboli.ttf",
    "Gabriola.ttf",
    "LHANDW.TTF",
    "BRADHITC.TTF",
    "ITCKRIST.TTF",
    "KalamRegular.ttf",
    "Caveat-Regular.ttf",
    "PatrickHand-Regular.ttf",
    "IndieFlower-Regular.ttf",
)


def default_font_dirs() -> list[Path]:
    if sys.platform == "win32":
        return [Path("C:/Windows/Fonts")]
    if sys.platform == "darwin":
        return [Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    return [Path("/usr/share/fonts"), Path.home() / ".local/share/fonts"]


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _glyph_mask(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray:
    image = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(image).text((16, 16), char, font=font, fill=255)
    return np.asarray(image)


class FontLibrary:
    def __init__(self, font_dirs: list[Path] | None = None) -> None:
        names = {name.lower() for name in CANDIDATE_FONTS}
        found: dict[str, Path] = {}
        for directory in font_dirs or default_font_dirs():
            if directory.is_dir():
                for path in directory.rglob("*"):
                    if path.suffix.lower() in (".ttf", ".otf") and path.name.lower() in names:
                        found.setdefault(path.name.lower(), path)
        self.paths = sorted(found.values(), key=lambda p: p.name.lower())
        self._coverage: dict[tuple[Path, str], bool] = {}

    def supports(self, path: Path, char: str) -> bool:
        key = (path, char)
        if key not in self._coverage:
            font = _font(str(path), 48)
            missing = _glyph_mask(font, "\uE000")  # private-use code point: renders .notdef
            self._coverage[key] = char.isspace() or not np.array_equal(
                _glyph_mask(font, char), missing
            )
        return self._coverage[key]

    def covering(self, text: str) -> list[Path]:
        return [path for path in self.paths if all(self.supports(path, c) for c in set(text))]


class DigitBank:
    """Handwritten digits from one dataset directory.

    Supported layouts: MNIST or EMNIST IDX files (optionally gzipped; EMNIST
    images are stored transposed), Hugging Face image parquet files named
    `train*.parquet` / `test*.parquet` (e.g. USPS), and folders of scanned pen
    digits named `<digit>-*.png` (CVL single digits: `train/` and `valid/`,
    both used for the train split). Images are stored as white ink on black. The train and test splits of these datasets come from different
    writers, so training data should use `split="train"` and evaluation data
    `split="test"`.
    """

    SPLIT_KEYS = {"train": ("train",), "test": ("t10k", "test")}
    FOLDERS = {"train": ("train", "valid"), "test": ("test",)}

    def __init__(self, directory: Path, split: str = "train") -> None:
        if split not in self.SPLIT_KEYS:
            raise ValueError(f"unknown digit split {split!r}")
        folders = [directory / name for name in self.FOLDERS[split] if (directory / name).is_dir()]
        parquet = self._matching(directory, split, ".parquet")
        if folders:
            images, labels = self._read_folders(folders)
        elif parquet:
            images, labels = self._read_parquet(parquet)
        else:
            idx = [p for p in self._matching(directory, split, "") if "idx" in p.name]
            image_files = [p for p in idx if "images" in p.name]
            label_files = [p for p in idx if "labels" in p.name]
            if not image_files or not label_files:
                raise FileNotFoundError(f"no {split} digit files in {directory}")
            images, labels = self._read(image_files[0]), self._read(label_files[0])
            if "emnist" in image_files[0].name:
                images = images.transpose(0, 2, 1)
        if len(images) != labels.shape[0]:
            raise ValueError("digit image and label counts differ")
        self.name = directory.name
        self.split = split
        self.images = images
        self.by_digit = {d: np.flatnonzero(labels == d) for d in range(10)}

    @classmethod
    def _matching(cls, directory: Path, split: str, suffix: str) -> list[Path]:
        return [
            path
            for path in sorted(directory.iterdir())
            if path.name.endswith(suffix) and any(key in path.name for key in cls.SPLIT_KEYS[split])
        ]

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        magic, count = struct.unpack(">II", data[:8])
        dims = magic & 0xFF
        shape = (count, *struct.unpack(f">{dims - 1}I", data[8 : 4 + 4 * dims]))
        return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)

    @staticmethod
    def _read_parquet(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        import io

        import pyarrow.parquet as pq

        images, labels = [], []
        for path in paths:
            table = pq.read_table(path, columns=["image", "label"]).to_pylist()
            for row in table:
                images.append(np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")))
                labels.append(row["label"])
        size = max(max(i.shape) for i in images)
        padded = np.zeros((len(images), size, size), np.uint8)
        for k, image in enumerate(images):
            padded[k, : image.shape[0], : image.shape[1]] = image
        return padded, np.asarray(labels)

    @staticmethod
    def _read_folders(folders: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
        """Scanned dark ink on light paper, turned into an ink mask."""
        images, labels = [], []
        for folder in folders:
            for path in sorted(folder.glob("*.png")):
                gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
                paper = np.percentile(gray, 90)
                ink = np.clip((paper - gray) / max(paper - gray.min(), 1.0), 0, 1)
                images.append((ink * 255).astype(np.uint8))
                labels.append(int(path.name.split("-")[0]))
        return images, np.asarray(labels)

    def sample(self, digit: int, rng: np.random.Generator) -> np.ndarray:
        image = self.images[rng.choice(self.by_digit[digit])].astype(np.float32) / 255
        ys, xs = np.nonzero(image > 0.1)
        return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


class DigitSources:
    """Several digit datasets; each digit is drawn from a randomly chosen one."""

    def __init__(self, directories: list[Path], split: str = "train") -> None:
        if not directories:
            raise ValueError("no digit directories")
        self.banks = [DigitBank(directory, split) for directory in directories]
        self.split = split
        self.name = "+".join(bank.name for bank in self.banks)

    def sample(self, digit: int, rng: np.random.Generator) -> np.ndarray:
        return self.banks[rng.integers(len(self.banks))].sample(digit, rng)


@dataclass(frozen=True)
class WriterStyle:
    """Consistent habits of one writer across a sheet."""

    font_path: Path
    ink: tuple[float, float, float]
    slant: float
    stroke: int
    rotation_jitter: float
    spacing: float
    scale_jitter: float


INKS = {
    "blue": (0.10, 0.20, 0.62),
    "dark_blue": (0.06, 0.10, 0.38),
    "black": (0.08, 0.08, 0.10),
    "red": (0.78, 0.10, 0.12),
}


def sample_style(fonts: list[Path], inks: list[str], rng: np.random.Generator) -> WriterStyle:
    if not fonts:
        raise ValueError("no handwriting font covers the requested text")
    ink = np.clip(np.array(INKS[inks[rng.integers(len(inks))]]) + rng.normal(0, 0.03, 3), 0, 1)
    return WriterStyle(
        font_path=fonts[rng.integers(len(fonts))],
        ink=tuple(float(c) for c in ink),
        slant=float(rng.normal(0.08, 0.12)),
        stroke=int(rng.choice([-1, 0, 0, 1, 1, 2])),
        rotation_jitter=float(rng.uniform(1.5, 7)),
        spacing=float(rng.uniform(-0.06, 0.12)),
        scale_jitter=float(rng.uniform(0.03, 0.12)),
    )


def _rotate(glyph: np.ndarray, degrees: float) -> np.ndarray:
    h, w = glyph.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(glyph, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)


def _elastic(mask: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    h, w = mask.shape
    sigma = max(h, w) * 0.08
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    scale = strength * h / (np.abs(dx).max() + 1e-6)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, grid_x + dx * scale, grid_y + dy * scale, cv2.INTER_LINEAR)


def render_text(
    text: str,
    style: WriterStyle,
    height: int,
    rng: np.random.Generator,
    digits: DigitBank | DigitSources | None = None,
) -> np.ndarray:
    """Returns a float32 alpha mask (1 = full ink) whose cap height is about `height` px."""

    size = max(8, int(height * 1.35))
    font = _font(str(style.font_path), size)
    ascent, descent = font.getmetrics()
    canvas_h = ascent + descent + size
    baseline = ascent + size // 2
    pieces: list[tuple[np.ndarray, int, int]] = []  # glyph, x, top
    x = 0
    for char in text:
        if char == " ":
            x += int(font.getlength(" ") * rng.uniform(1.0, 1.6))
            continue
        scale = 1 + rng.normal(0, style.scale_jitter)
        if digits is not None and char.isdigit():
            glyph = digits.sample(int(char), rng)
            target_h = max(4, int(height * scale))
            glyph = cv2.resize(glyph, (max(2, int(glyph.shape[1] * target_h / glyph.shape[0])), target_h))
            glyph = _rotate(glyph, rng.normal(0, style.rotation_jitter))
            top = baseline - glyph.shape[0] + int(rng.normal(0, height * 0.04))
            bearing, advance = 0, glyph.shape[1] + height * 0.12
        else:
            # Keep the font's own side bearing and advance so letters keep
            # their natural rhythm; only small jitter is added on top.
            origin = size // 2
            image = Image.new("L", (size * 2, canvas_h), 0)
            ImageDraw.Draw(image).text((origin, baseline), char, font=font, fill=255, anchor="ls")
            glyph = np.asarray(image, dtype=np.float32) / 255
            cols = np.flatnonzero(glyph.max(axis=0) > 0.05)
            if cols.size == 0:
                x += int(font.getlength(char) or size * 0.3)
                continue
            bearing = int((cols[0] - origin) * scale)
            advance = font.getlength(char) * scale
            glyph = glyph[:, cols[0] : cols[-1] + 1]
            if abs(scale - 1) > 0.01:
                glyph = cv2.resize(glyph, (max(2, int(glyph.shape[1] * scale)), max(2, int(canvas_h * scale))))
            glyph = _rotate(glyph, rng.normal(0, style.rotation_jitter))
            top = int(baseline - glyph.shape[0] / 2 - (baseline - canvas_h / 2) * scale)
            top += int(rng.normal(0, height * 0.04))
        pieces.append((glyph, max(0, x + bearing), top))
        x += int(advance * (1 + style.spacing) + rng.normal(0, size * 0.025))

    if not pieces:
        return np.zeros((height, 1), dtype=np.float32)
    min_top = min(top for _, _, top in pieces)
    width = max(px + g.shape[1] for g, px, _ in pieces) + 4
    total_h = max(top + g.shape[0] for g, _, top in pieces) - min_top + 4
    mask = np.zeros((total_h, width), dtype=np.float32)
    for glyph, px, top in pieces:
        y0 = top - min_top + 2
        region = mask[y0 : y0 + glyph.shape[0], px + 2 : px + 2 + glyph.shape[1]]
        np.maximum(region, glyph[: region.shape[0], : region.shape[1]], out=region)

    # Slant (shear), then elastic wobble and pen-width variation.
    pad = int(abs(style.slant) * total_h) + 2
    mask = cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    shear = np.float32([[1, -style.slant, style.slant * total_h / 2], [0, 1, 0]])
    mask = cv2.warpAffine(mask, shear, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
    mask = _elastic(mask, rng, strength=rng.uniform(0.01, 0.035))
    if style.stroke:
        kernel = np.ones((abs(style.stroke) + 1, abs(style.stroke) + 1), np.uint8)
        mask = (cv2.dilate if style.stroke > 0 else cv2.erode)(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.3, 0.7))
    rows, cols = np.flatnonzero(mask.max(axis=1) > 0.05), np.flatnonzero(mask.max(axis=0) > 0.05)
    if rows.size == 0:
        return np.zeros((height, 1), dtype=np.float32)
    mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    pressure = 0.75 + 0.25 * cv2.GaussianBlur(
        rng.uniform(0, 1, mask.shape).astype(np.float32), (0, 0), max(2.0, height * 0.3)
    )
    return np.clip(mask * pressure / max(pressure.max(), 1e-6) * rng.uniform(0.85, 1.0), 0, 1)


def strike_through(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crosses out written text the way a corrected score is crossed out."""

    h, w = mask.shape
    out = mask.copy()
    thickness = max(1, int(h * rng.uniform(0.06, 0.12)))
    style = rng.choice(["line", "double", "scribble"])
    if style == "scribble":
        points = [(int(rng.uniform(0, w * 0.15)), int(rng.uniform(h * 0.2, h * 0.8)))]
        for i in range(int(rng.integers(5, 9))):
            points.append((int(w * (i + 1) / 8 + rng.normal(0, w * 0.04)), int(h * (0.15 if i % 2 else 0.85))))
        cv2.polylines(out, [np.array(points, np.int32)], False, 1.0, thickness, cv2.LINE_AA)
    else:
        for offset in [0.0] if style == "line" else [-0.12, 0.12]:
            y0 = int(h * (0.5 + offset + rng.normal(0, 0.08)))
            y1 = int(h * (0.5 + offset + rng.normal(0, 0.15)))
            cv2.line(out, (0, y0), (w - 1, y1), 1.0, thickness, cv2.LINE_AA)
    return out
