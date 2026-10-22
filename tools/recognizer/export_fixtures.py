#!/usr/bin/env python3
"""Write parity fixtures for the Rust recognizer.

Picks a few crops of every kind from a synthetic split, stores them as raw RGB
bytes (no image decoder needed in Rust tests) and records what the Python
reference produces for them with the exported ONNX model (expected.tsv):
preprocessed tensor width and sum, decoded text and confidence.

    python tools/recognizer/export_fixtures.py --model models/digit-crnn-v4.onnx \
        --data data/synthetic/eval-v2/shard-2 --out crates/exam_infrastructure/tests/fixtures/recognizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import FIELD_ALPHABETS, decode, preprocess  # noqa: E402
from train import load_samples, read_rgb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=3)
    args = parser.parse_args(argv)

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.model))
    samples = load_samples([args.data])
    chosen = []
    for predicate in (
        lambda s: s.kind == "score" and not s.corrected and not s.blank,
        lambda s: s.kind == "score" and s.corrected,
        lambda s: s.blank,
        lambda s: s.kind == "total" and not s.blank,
        lambda s: s.kind == "number",
    ):
        chosen += [s for s in samples if predicate(s)][: args.per_kind]

    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, sample in enumerate(chosen):
        rgb = read_rgb(sample.path)
        tensor = preprocess(rgb)
        probabilities = session.run(None, {"image": tensor[None]})[0][0]
        text, confidence = decode(probabilities, FIELD_ALPHABETS[sample.kind])
        name = f"{index:02d}-{sample.kind}.rgb"
        (args.out / name).write_bytes(np.ascontiguousarray(rgb).tobytes())
        cases.append(
            {
                "file": name,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "kind": sample.kind,
                "label": sample.text,
                "tensor_width": int(tensor.shape[2]),
                "tensor_sum": float(tensor.astype(np.float64).sum()),
                "text": text,
                "confidence": confidence,
            }
        )
    # Tab-separated so the Rust test needs no JSON dependency.
    columns = ["file", "width", "height", "kind", "label", "tensor_width", "tensor_sum", "text", "confidence"]
    lines = ["\t".join(columns)] + ["\t".join(str(case[c]) for c in columns) for case in cases]
    (args.out / "expected.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    agree = sum(c["text"] == c["label"] for c in cases)
    print(f"wrote {len(cases)} fixtures to {args.out} ({agree} decoded exactly as labelled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
