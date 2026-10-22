#!/usr/bin/env python3
"""Train the digit cell recognizer on synthetic crops and export it to ONNX.

    python tools/recognizer/train.py \
        --train data/synthetic/train/shard-* \
        --validation data/synthetic/eval/shard-1 --test data/synthetic/eval/shard-2 \
        --out models --name digit-crnn-v4

The validation split picks the best epoch and the review threshold; the test
split is only scored once at the end with that threshold, so the reported
numbers are not tuned on the data they describe.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    ALPHABET,
    FIELD_ALPHABETS,
    DigitCRNN,
    ExportedRecognizer,
    decode,
    encode,
    label_text,
    preprocess,
)

TRAINER_VERSION = "1.0.0"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


@dataclass(frozen=True)
class Sample:
    path: Path
    kind: str  # score | total | number
    text: str
    corrected: bool
    blank: bool


def field_kind(field: str) -> str | None:
    if field.startswith("question:"):
        return "score"
    return {"declared_total": "total", "student.number": "number"}.get(field)


def load_samples(directories: list[Path]) -> list[Sample]:
    samples = []
    for directory in directories:
        with (directory / "crops" / "labels.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                kind = field_kind(row["field"])
                if kind is None:
                    continue
                corrected = row["corrected"] == "1"
                samples.append(
                    Sample(directory / row["crop"], kind, label_text(row["text"], corrected), corrected, row["blank"] == "1")
                )
    if not samples:
        raise SystemExit(f"no digit crops found in {directories}")
    return samples


def load_real_strings(directories: list[Path]) -> list[Sample]:
    """Scanned real digit strings named `<digits>-*.png` (e.g. CVL), read as numbers."""
    return [
        Sample(path, "number", path.name.split("-")[0], False, False)
        for directory in directories
        for path in sorted(directory.glob("*.png"))
        if path.name.split("-")[0].isdigit()
    ]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def augment(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Training-only photometric and small geometric noise on top of the generator's."""
    h, w = rgb.shape[:2]
    if rng.random() < 0.5:  # imperfect cell localisation
        dx0, dx1 = int(w * rng.uniform(-0.04, 0.05)), int(w * rng.uniform(-0.04, 0.05))
        dy0, dy1 = int(h * rng.uniform(-0.06, 0.08)), int(h * rng.uniform(-0.06, 0.08))
        rgb = cv2.copyMakeBorder(rgb, max(0, -dy0), max(0, -dy1), max(0, -dx0), max(0, -dx1), cv2.BORDER_REPLICATE)
        rgb = rgb[max(0, dy0) : rgb.shape[0] - max(0, dy1), max(0, dx0) : rgb.shape[1] - max(0, dx1)]
    if rng.random() < 0.3:  # pen width: thicker (min filter) or thinner (max filter) dark strokes
        kernel = np.ones((2, 2), np.uint8)
        rgb = cv2.erode(rgb, kernel) if rng.random() < 0.5 else cv2.dilate(rgb, kernel)
    if rng.random() < 0.3:
        h, w = rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), 1.0)
        rgb = cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    image = rgb.astype(np.float32)
    image = (image - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.1:
        image = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)
    image += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 6), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def batches(samples: list[Sample], batch_size: int, rng: random.Random, shuffle: bool) -> list[list[Sample]]:
    """Groups crops of similar width so little compute is spent on padding."""
    order = samples[:]
    if shuffle:
        rng.shuffle(order)
    chunk = batch_size * 32
    grouped = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start : start + chunk], key=lambda s: s.kind)
        grouped += [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(grouped)
    return grouped


def stack(tensors: list[np.ndarray]) -> torch.Tensor:
    width = max(t.shape[2] for t in tensors)
    images = np.ones((len(tensors), 3, tensors[0].shape[1], width), np.float32)  # white padding
    for i, tensor in enumerate(tensors):
        images[i, :, :, : tensor.shape[2]] = tensor
    return torch.from_numpy(images)


def collate(batch: list[Sample], rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = stack([preprocess(augment(read_rgb(s.path), rng)) for s in batch])
    targets = [encode(s.text) for s in batch]
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return images, flat, lengths


class BatchDataset(torch.utils.data.Dataset):
    """Pre-grouped batches, read and augmented in DataLoader worker processes."""

    def __init__(self, groups: list[list[Sample]], seed: int) -> None:
        self.groups = groups
        self.seed = seed

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return collate(self.groups[index], random.Random(self.seed * 1_000_003 + index))


def _single_threaded_worker(_: int) -> None:
    """Each worker uses one core; many workers in parallel beat threads fighting."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict(model: nn.Module, samples: list[Sample], batch_size: int = 128) -> list[tuple[str, float]]:
    """Decodes every sample over its own frames only (padding is ignored)."""
    model.eval()
    device = model_device(model)
    results: list[tuple[str, float]] = [("", 0.0)] * len(samples)
    order = sorted(range(len(samples)), key=lambda i: samples[i].kind)  # similar widths together
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        tensors = [preprocess(read_rgb(samples[i].path)) for i in chunk]
        probabilities = model(stack(tensors).to(device)).softmax(dim=2).cpu().numpy()
        for index, tensor, matrix in zip(chunk, tensors, probabilities):
            results[index] = decode(matrix[: tensor.shape[2] // 4], FIELD_ALPHABETS[samples[index].kind])
    return results


def character_errors(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score(samples: list[Sample], predictions: list[tuple[str, float]], threshold: float | None = None) -> dict:
    def rate(hits: int, total: int) -> float | None:
        return round(hits / total, 4) if total else None

    report: dict = {"count": len(samples)}
    for kind in ("score", "total", "number", "all"):
        pairs = [(s, p) for s, p in zip(samples, predictions) if kind in ("all", s.kind)]
        exact = sum(s.text == p[0] for s, p in pairs)
        chars = sum(len(s.text) for s, _ in pairs)
        report[kind] = {
            "count": len(pairs),
            "exact_match": rate(exact, len(pairs)),
            "character_error_rate": rate(sum(character_errors(s.text, p[0]) for s, p in pairs), chars),
        }
    corrected = [(s, p) for s, p in zip(samples, predictions) if s.corrected]
    clean = [(s, p) for s, p in zip(samples, predictions) if not s.corrected]
    report["correction_marker"] = {
        "recall": rate(sum(p[0].startswith("~") for _, p in corrected), len(corrected)),
        "false_alarm_rate": rate(sum("~" in p[0] for _, p in clean), len(clean)),
    }
    blanks = [(s, p) for s, p in zip(samples, predictions) if s.blank]
    report["blank_detection"] = {"accuracy": rate(sum(p[0] == "" for _, p in blanks), len(blanks)), "count": len(blanks)}
    curve = []
    for value in THRESHOLDS if threshold is None else (threshold,):
        accepted = [(s, p) for s, p in zip(samples, predictions) if p[1] >= value]
        curve.append(
            {
                "threshold": value,
                "coverage": rate(len(accepted), len(samples)),
                "accepted_exact_match": rate(sum(s.text == p[0] for s, p in accepted), len(accepted)),
                "accepted_errors": sum(s.text != p[0] for s, p in accepted),
            }
        )
    report["confidence"] = curve
    return report


def choose_threshold(curve: list[dict], target: float, minimum_accepted: int, total: int) -> float | None:
    """Lowest threshold whose accepted fields reach the target accuracy."""
    for point in curve:
        accepted = round((point["coverage"] or 0) * total)
        if accepted >= minimum_accepted and (point["accepted_exact_match"] or 0) >= target:
            return point["threshold"]
    return None


def export_onnx(model: DigitCRNN, path: Path) -> None:
    exported = ExportedRecognizer(model).cpu().eval()
    dummy = torch.zeros(1, 3, 32, 128)
    torch.onnx.export(
        exported,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch", 3: "width"}, "probabilities": {0: "batch", 1: "time"}},
        opset_version=17,
        dynamo=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--real-strings", nargs="*", default=[], help="real digit-string images added to training")
    parser.add_argument("--real-repeat", type=int, default=1, help="times each real string is used per epoch")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--name", default="digit-crnn-v4")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto (CUDA when available), cuda or cpu")
    parser.add_argument("--workers", type=int, default=6, help="processes that read and augment crops")
    parser.add_argument("--width", type=float, default=1.0, help="channel multiplier of the model")
    parser.add_argument("--target-accuracy", type=float, default=0.99, help="accuracy required for auto-accepted fields")
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--evaluate-only", action="store_true", help="re-score and re-export the saved checkpoint")
    args = parser.parse_args(argv)

    expand = lambda patterns: sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})  # noqa: E731
    train, validation, test = (load_samples(expand(p)) for p in (args.train, args.validation, args.test))
    real = load_real_strings(expand(args.real_strings))
    train += real * args.real_repeat
    print(f"train {len(train)} ({len(real)} real strings)  validation {len(validation)}  test {len(test)} crops", flush=True)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    print(f"device {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    model = DigitCRNN(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(batches(train, args.batch_size, random.Random(0), shuffle=False))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / f"{args.name}.pt"

    best, history = -1.0, []
    if args.evaluate_only:
        previous = args.out / f"{args.name}.metrics.json"
        history = json.loads(previous.read_text(encoding="utf-8"))["history"] if previous.exists() else []
    for epoch in range(1, 0 if args.evaluate_only else args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        loader = torch.utils.data.DataLoader(
            BatchDataset(batches(train, args.batch_size, rng, shuffle=True), rng.getrandbits(31)),
            batch_size=None,
            num_workers=args.workers,
            worker_init_fn=_single_threaded_worker,
            pin_memory=device.type == "cuda",
        )
        for images, targets, lengths in loader:
            images, targets, lengths = images.to(device), targets.to(device), lengths.to(device)
            log_probs = model(images).log_softmax(dim=2).permute(1, 0, 2)  # (time, batch, classes)
            frames = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
            loss = ctc(log_probs, targets, frames, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        metrics = score(validation, predict(model, validation))
        accuracy = metrics["all"]["exact_match"]
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "validation_exact_match": accuracy})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  validation exact {accuracy:.4f}  ({time.time() - started:.0f}s)", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "alphabet": ALPHABET, "width": args.width},
                checkpoint,
            )

    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)["model_state"])
    validation_report = score(validation, predict(model, validation))
    threshold = choose_threshold(validation_report["confidence"], args.target_accuracy, args.min_accepted, len(validation))
    test_report = score(test, predict(model, test))
    test_at_threshold = score(test, predict(model, test), threshold)["confidence"][0] if threshold else None

    onnx_path = args.out / f"{args.name}.onnx"
    export_onnx(model, onnx_path)
    report = {
        "model": args.name,
        "trainer_version": TRAINER_VERSION,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_bytes": onnx_path.stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "alphabet": ALPHABET,
        "data": {
            "train": [str(p) for p in expand(args.train)],
            "real_strings": [str(p) for p in expand(args.real_strings)],
            "validation": [str(p) for p in expand(args.validation)],
            "test": [str(p) for p in expand(args.test)],
            "corpus": "synthetic İnönü sheets; NOT a real-sheet benchmark",
            "digit_sources": sorted(
                {
                    json.loads((p / "manifest.json").read_text(encoding="utf-8")).get("digit_source", "?")
                    for p in expand(args.train + args.validation + args.test)
                    if (p / "manifest.json").exists()
                }
            ),
        },
        "history": history,
        "selected_threshold": {
            "value": threshold,
            "target_accuracy": args.target_accuracy,
            "chosen_on": "validation",
            "test_at_threshold": test_at_threshold,
        },
        "validation": validation_report,
        "test": test_report,
    }
    (args.out / f"{args.name}.metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": test_report["all"], "threshold": report["selected_threshold"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
