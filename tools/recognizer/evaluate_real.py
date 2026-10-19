#!/usr/bin/env python3
"""Score a trained recognizer on real handwritten digit strings it never saw.

    python tools/recognizer/evaluate_real.py --model models/digit-crnn-v2.pt \
        --cvl data/raw/cvl/strings-eval/cvl-strings-eval --out models/digit-crnn-v2.real.json

The CVL Digit Strings evaluation set (ICDAR 2013, TU Wien; CC BY-NC 4.0) holds
6,698 scanned strings of 5-7 digits by about 300 writers. File names start with
the true string and the second field is the writer ("102246-0001-21.png").
`--exclude-writers-from` drops every writer that appears in the given training
folders, so the score covers only people whose handwriting the model never saw.
Strings are decoded as the student-number field (digits only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DigitCRNN  # noqa: E402
from train import Sample, predict, score  # noqa: E402


def writer(path: Path) -> str:
    return path.name.split("-")[1]


def cvl_samples(directory: Path, excluded_writers: set[str]) -> list[Sample]:
    samples = [
        Sample(path, "number", path.name.split("-")[0], False, False)
        for path in sorted(directory.glob("*.png"))
        if writer(path) not in excluded_writers
    ]
    if not samples:
        raise SystemExit(f"no CVL images in {directory}")
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--cvl", type=Path, required=True, help="CVL digit strings directory")
    parser.add_argument("--exclude-writers-from", type=Path, nargs="*", default=[], help="training folders")
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    model = DigitCRNN()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location="cpu")["model_state"])
    if torch.cuda.is_available():
        model = model.cuda()
    excluded = {writer(p) for folder in args.exclude_writers_from for p in folder.rglob("*.png")}
    samples = cvl_samples(args.cvl, excluded)
    predictions = predict(model, samples)
    report = score(samples, predictions)
    by_length: dict[int, list[bool]] = {}
    for sample, (text, _) in zip(samples, predictions):
        by_length.setdefault(len(sample.text), []).append(sample.text == text)
    summary = {
        "model": args.model.name,
        "corpus": "CVL Digit Strings eval (real handwriting)",
        "count": len(samples),
        "writers": len({writer(s.path) for s in samples}),
        "excluded_writers": len(excluded),
        "exact_match": report["number"]["exact_match"],
        "character_error_rate": report["number"]["character_error_rate"],
        "exact_match_by_length": {n: round(sum(v) / len(v), 4) for n, v in sorted(by_length.items())},
        "confidence": report["confidence"],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
