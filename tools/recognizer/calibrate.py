#!/usr/bin/env python3
"""How far the recognizer's confidence can be trusted, field by field.

The reader accepts a cell when its confidence reaches a threshold and sends
the rest to the instructor. After the reject-option work on handwriting
verification (confidence scoring as a post-processor deciding what to accept,
e.g. the IBM TPAMI paper on off-line handwritten character verification), the
threshold is chosen from measured error, not set by hand: for each field kind
the lowest threshold whose accepted cells are right at least `--target` of the
time, with how many cells it accepts (coverage) and the calibration error (how
far confidence is from the observed accuracy).

Student numbers are measured on real handwriting (CVL writers the model never
saw); score and total cells on synthetic sheets with real digit strokes.

    python tools/recognizer/calibrate.py --model models/digit-crnn-v4.pt \\
        --cvl data/raw/cvl/strings-eval/cvl-strings-eval \\
        --exclude-writers-from data/raw/cvl/strings-train data/raw/cvl/cvl-single \\
        --synthetic data/synthetic/eval-v2/shard-1 --out models/digit-crnn-v4.calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DigitCRNN  # noqa: E402
from train import Sample, load_samples, predict  # noqa: E402


def writer(path: Path) -> str:
    return path.name.split("-")[1]


def curve(right: np.ndarray, confidence: np.ndarray, target: float) -> dict:
    """Accuracy and coverage of the accepted cells at each threshold."""
    rows = []
    # Finer towards 1, where the confident readings are.
    thresholds = list(np.round(np.arange(0.30, 0.95, 0.05), 3)) + [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
    for threshold in thresholds:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": round(count / len(right), 4),
                "accuracy": round(float(right[accepted].mean()), 4) if count else None,
            }
        )
    reaching = [r for r in rows if r["accuracy"] is not None and r["accuracy"] >= target and r["coverage"] >= 0.05]
    # Expected calibration error over ten confidence bins.
    bins = np.clip((confidence * 10).astype(int), 0, 9)
    ece = sum(
        abs(right[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()
    )
    return {
        "count": int(len(right)),
        "accuracy": round(float(right.mean()), 4),
        "calibration_error": round(float(ece), 4),
        "chosen": reaching[0] if reaching else None,
        "curve": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cvl", type=Path, required=True)
    parser.add_argument("--exclude-writers-from", type=Path, nargs="*", default=[])
    parser.add_argument("--held-out-only", action="store_true", help="only the CVL writers self_train.py scores on")
    parser.add_argument("--synthetic", type=Path, nargs="*", default=[])
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, weights_only=True, map_location=device)
    model = DigitCRNN(width=checkpoint.get("width", 1.0)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    excluded = {writer(p) for folder in args.exclude_writers_from for p in folder.rglob("*.png")}
    held_out = lambda p: int(hashlib.sha1(writer(p).encode()).hexdigest(), 16) % 2 == 1  # noqa: E731
    numbers = [
        Sample(p, "number", p.name.split("-")[0], False, False)
        for p in sorted(args.cvl.glob("*.png"))
        if writer(p) not in excluded and (held_out(p) or not args.held_out_only)
    ]
    groups: dict[str, list[Sample]] = {"number": numbers}
    for sample in load_samples(args.synthetic) if args.synthetic else []:
        if not sample.blank:
            groups.setdefault(sample.kind, []).append(sample)

    report = {"model": args.model.name, "target_accuracy": args.target}
    for kind, samples in groups.items():
        predictions = predict(model, samples, batch_size=1)
        right = np.array([s.text == text for s, (text, _) in zip(samples, predictions)], dtype=float)
        confidence = np.array([c for _, c in predictions], dtype=float)
        report[kind] = curve(right, confidence, args.target)
        chosen = report[kind]["chosen"]
        print(
            f"{kind:7s} n={len(samples):5d} accuracy {report[kind]['accuracy']:.3f}  "
            f"calibration error {report[kind]['calibration_error']:.3f}  "
            + (f"threshold {chosen['threshold']} -> accepts {chosen['coverage']:.1%} at {chosen['accuracy']:.1%}" if chosen else "target not reached"),
            flush=True,
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
