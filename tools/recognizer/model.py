"""Cell recognizer shared by training, export and tests.

The preprocessing and decoding here are the contract the Rust runtime
re-implements (`exam_infrastructure::recognition`); change them together.

- Input: an RGB crop of one field (score cell, `Not:` total or student number).
- Preprocess: bilinear resize (pixel-centre aligned, edge-clamped) to
  INPUT_HEIGHT, width scaled by the same factor (rounded half up) and clamped to
  [MIN_WIDTH, MAX_WIDTH]; pad on the right with white to a multiple of
  WIDTH_MULTIPLE; scale to [-1, 1]; NCHW float32.
- Output: per-frame class probabilities, shape (batch, time, classes);
  class 0 is the CTC blank, then ALPHABET.
- Decode: greedy CTC (argmax, collapse repeats, drop blanks) over the blank
  and the characters allowed for the field (FIELD_ALPHABETS); other classes
  are ignored, not renormalised. Confidence is the minimum over frames of the
  winning probability, so one uncertain frame (including an uncertain blank)
  lowers the whole cell.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

ALPHABET = "0123456789.~"  # "." = any decimal separator, "~" = a crossed-out correction is present
# Characters each field may contain; decoding ignores the others (a student
# number never has a decimal point or a correction marker).
FIELD_ALPHABETS = {"score": ALPHABET, "total": ALPHABET, "number": "0123456789"}
BLANK = 0
INPUT_HEIGHT = 32
MIN_WIDTH = 16
MAX_WIDTH = 512
WIDTH_MULTIPLE = 4


def label_text(written: str, corrected: bool) -> str:
    """Training target for a crop: decimal commas become '.', corrections get a '~' prefix."""
    text = written.replace(",", ".")
    return f"~{text}" if corrected else text


def encode(text: str) -> list[int]:
    return [ALPHABET.index(char) + 1 for char in text]


def resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pixel-centre aligned bilinear resize with edge clamping (no antialiasing)."""
    src_h, src_w = image.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * src_h / height - 0.5, 0, src_h - 1)
    xs = np.clip((np.arange(width) + 0.5) * src_w / width - 0.5, 0, src_w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, src_h - 1), np.minimum(x0 + 1, src_w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    image = image.astype(np.float32)
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def target_width(src_h: int, src_w: int) -> int:
    # Round half up explicitly: Python's round() rounds half to even, which the
    # Rust runtime would not reproduce.
    scaled = math.floor(src_w * INPUT_HEIGHT / src_h + 0.5)
    return int(min(MAX_WIDTH, max(MIN_WIDTH, scaled)))


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB crop -> float32 3 x INPUT_HEIGHT x W tensor in [-1, 1]."""
    width = target_width(*rgb.shape[:2])
    resized = resize_bilinear(rgb, INPUT_HEIGHT, width)
    padded_width = -(-width // WIDTH_MULTIPLE) * WIDTH_MULTIPLE
    if padded_width > width:
        resized = np.pad(resized, ((0, 0), (0, padded_width - width), (0, 0)), constant_values=255.0)
    return (resized.transpose(2, 0, 1) / 127.5 - 1.0).astype(np.float32)


def decode(probabilities: np.ndarray, allowed: str = ALPHABET) -> tuple[str, float]:
    """Greedy CTC decode of one (time, classes) probability matrix."""
    if allowed != ALPHABET:
        keep = [BLANK] + [ALPHABET.index(char) + 1 for char in allowed]
        masked = np.zeros_like(probabilities)
        masked[:, keep] = probabilities[:, keep]
        probabilities = masked
    best = probabilities.argmax(axis=1)
    confidence = float(probabilities.max(axis=1).min()) if len(best) else 0.0
    chars, previous = [], BLANK
    for index in best:
        if index != previous and index != BLANK:
            chars.append(ALPHABET[index - 1])
        previous = index
    return "".join(chars), confidence


def _block(inputs: int, outputs: int) -> list[nn.Module]:
    return [nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)]


class DigitCRNN(nn.Module):
    """Fully convolutional CTC recognizer: height 32 -> 1, width / 4 time steps."""

    def __init__(self, classes: int = len(ALPHABET) + 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_block(3, 32), nn.MaxPool2d(2),
            *_block(32, 64), nn.MaxPool2d(2),
            *_block(64, 128), *_block(128, 128), nn.MaxPool2d((2, 1)),
            *_block(128, 160), nn.MaxPool2d((2, 1)),
            nn.Conv2d(160, 160, (2, 1), bias=False), nn.BatchNorm2d(160), nn.ReLU(inplace=True),
        )
        self.sequence = nn.Sequential(
            nn.Conv1d(160, 160, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(160, 160, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(160, classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, time, classes)."""
        features = self.features(images).squeeze(2)
        return self.sequence(features).permute(0, 2, 1)


class ExportedRecognizer(nn.Module):
    """What ships: probabilities instead of logits, so runtimes only need argmax/min."""

    def __init__(self, model: DigitCRNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).softmax(dim=2)
