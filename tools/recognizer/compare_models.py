#!/usr/bin/env python3
"""Compare digit recognizers on real handwriting: our model and ready-made ones.

    .venv/Scripts/python tools/recognizer/compare_models.py \
        --cvl data/raw/cvl/strings-eval/cvl-strings-eval \
        --exclude-writers-from data/raw/cvl/strings-train data/raw/cvl/cvl-single \
        --models ours paddle trocr --out models/comparison.json

Every model reads the same CVL strings by writers it never saw and is decoded
the same way the app reads the student number: digits only. For CTC models the
other characters are masked out before greedy decoding. TrOCR generates freely
and non-digits are dropped from its text: it was trained on English text lines
and writes words for digit strings (e.g. "References and structures" for
136075); restricting it to digit tokens makes it worse (it repeats digits). Speed is per image, single images,
after warm-up; CPU unless noted.

Models
- ours: `models/digit-crnn-v4.onnx` (onnxruntime, CPU).
- paddle: PaddleOCR PP-OCRv4 mobile recognition (Chinese + Latin), the ONNX
  export shipped with `rapidocr-onnxruntime` (Apache-2.0).
- trocr: `microsoft/trocr-small-handwritten` (MIT), a 62M-parameter
  transformer, on the GPU when available. Not usable for live scanning on a
  phone; it shows what a large ready-made handwriting model achieves.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ALPHABET, decode, preprocess  # noqa: E402
from train import character_errors  # noqa: E402

DIGITS = "0123456789"


def writer(path: Path) -> str:
    return path.name.split("-")[1]


def samples(directory: Path, excluded: set[str], limit: int | None) -> list[tuple[Path, str]]:
    items = [
        (path, path.name.split("-")[0])
        for path in sorted(directory.glob("*.png"))
        if writer(path) not in excluded
    ]
    if limit:
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    return items


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Ours:
    name = "digit-crnn-v4 (bizim)"

    def __init__(self, root: Path) -> None:
        import onnxruntime as ort

        self.path = root / "models" / "digit-crnn-v4.onnx"
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])

    def read(self, rgb: np.ndarray) -> tuple[str, float]:
        tensor = preprocess(rgb)[None]
        probabilities = self.session.run(None, {"image": tensor})[0][0]
        return decode(probabilities, DIGITS)


class Paddle:
    name = "PaddleOCR PP-OCRv4 mobile"

    def __init__(self, root: Path) -> None:
        import onnxruntime as ort
        import rapidocr_onnxruntime

        self.path = Path(rapidocr_onnxruntime.__file__).parent / "models" / "ch_PP-OCRv4_rec_infer.onnx"
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        characters = self.session.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.characters = ["<blank>", *characters, " "]
        self.keep = np.array([0] + [i for i, c in enumerate(self.characters) if c in DIGITS])

    def read(self, rgb: np.ndarray) -> tuple[str, float]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        width = max(16, int(np.ceil(48 * w / h)))
        image = cv2.resize(bgr, (width, 48)).astype(np.float32) / 255.0
        image = ((image - 0.5) / 0.5).transpose(2, 0, 1)
        padded = np.zeros((3, 48, max(320, width)), np.float32)
        padded[:, :, :width] = image
        probabilities = self.session.run(None, {"x": padded[None]})[0][0]
        masked = probabilities[:, self.keep]
        best = masked.argmax(axis=1)
        text, previous = [], 0
        for index in best:
            if index != 0 and index != previous:
                text.append(self.characters[self.keep[index]])
            previous = index
        confidence = float(masked.max(axis=1).min()) if len(best) else 0.0
        return "".join(text), confidence


class TrOCR:
    name = "TrOCR small handwritten"

    def __init__(self, root: Path) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        repo = "microsoft/trocr-small-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(repo)
        self.model = VisionEncoderDecoderModel.from_pretrained(repo).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.path = None

    def read(self, rgb: np.ndarray) -> tuple[str, float]:
        from PIL import Image

        pixels = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        with self.torch.no_grad():
            out = self.model.generate(
                pixels.to(self.device),
                max_new_tokens=16,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        scores = [s.softmax(-1).max().item() for s in out.scores]
        return "".join(c for c in text if c in DIGITS), min(scores) if scores else 0.0


MODELS = {"ours": Ours, "paddle": Paddle, "trocr": TrOCR}


def size_mb(model) -> float | None:
    if getattr(model, "path", None):
        return round(model.path.stat().st_size / 2**20, 1)
    parameters = sum(p.numel() for p in model.model.parameters())
    return round(parameters * 4 / 2**20, 1)


def evaluate(model, items: list[tuple[Path, str]]) -> dict:
    images = [(read_rgb(path), truth) for path, truth in items]
    for rgb, _ in images[:5]:
        model.read(rgb)  # warm-up
    exact, errors, chars, times = 0, 0, 0, []
    confident = []
    for rgb, truth in images:
        started = time.perf_counter()
        text, confidence = model.read(rgb)
        times.append((time.perf_counter() - started) * 1000)
        exact += text == truth
        errors += character_errors(truth, text)
        chars += len(truth)
        confident.append((confidence, text == truth))
    high = [ok for c, ok in confident if c >= 0.9]
    return {
        "model": model.name,
        "count": len(images),
        "exact_match": round(exact / len(images), 4),
        "character_error_rate": round(errors / chars, 4),
        "ms_per_image_median": round(statistics.median(times), 2),
        "device": getattr(model, "device", "cpu"),
        "size_mb": size_mb(model),
        "confidence_0.9": {
            "coverage": round(len(high) / len(images), 4),
            "accuracy": round(sum(high) / len(high), 4) if high else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cvl", type=Path, required=True)
    parser.add_argument("--exclude-writers-from", type=Path, nargs="*", default=[])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--limit", type=int, help="evaluate an evenly spread subset")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    excluded = {writer(p) for folder in args.exclude_writers_from for p in folder.rglob("*.png")}
    items = samples(args.cvl, excluded, args.limit)
    print(f"{len(items)} strings by {len({writer(p) for p, _ in items})} unseen writers", flush=True)
    results = []
    for key in args.models:
        model = MODELS[key](root)
        result = evaluate(model, items)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
