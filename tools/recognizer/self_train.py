#!/usr/bin/env python3
"""Adapt the recognizer to new handwriting without labels (self-training).

After Wolf and Fink, "Self-Training of Handwritten Word Recognition for
Synthetic-to-Real Adaptation" (ICPR 2022): the model reads unlabeled crops of
the target handwriting, keeps the readings it is confident about as labels
(pseudo-labels) and is trained on them, cycle after cycle; each kept crop is
seen twice per cycle, weakly and strongly distorted, against the same label
(consistency, as in FixMatch). Synthetic crops with true labels are mixed in so
that what the target crops do not show (decimal points, corrections) is not
forgotten.

Unlabeled crops are either the instructor's (cells read in the app, exported
as `<anything>.png`, `--unlabeled`) or, to measure the method, half of the CVL
writers the model never saw (`--cvl`): their labels are ignored for training
and only used to score the other half.

    python tools/recognizer/self_train.py --model models/digit-crnn-v4.pt \\
        --cvl data/raw/cvl/strings-eval/cvl-strings-eval \\
        --exclude-writers-from data/raw/cvl/strings-train data/raw/cvl/cvl-single \\
        --synthetic "data/synthetic/train-v3/shard-*" --out models --name digit-crnn-v5-self

Measured on `digit-crnn-v4` (the other half of the unseen CVL writers, 85.6%
exact at the start): self-training does NOT help this model. With
pseudo-labels kept from confidence 0.99 (94% of them right) the score falls
cycle after cycle, to 48.0% after 8 cycles with the strong view and 80.4%
after 4 without it: the remaining errors are the hard ones and the wrong 6%
reinforce themselves. The paper starts from a weak synthetic-only model (27%
CER; ours is at 2.7%). Trained on the TRUE labels instead (`--oracle`, as the
instructor's confirmed or corrected cells would be), the same loop raises it
to 87.0% in 2 cycles. So adapt on checked cells, not on the model's own
readings. Batch-norm statistics are kept frozen: batches mixing wide strings
and narrow cells shifted them and hurt even training on true labels.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import DigitCRNN, encode, preprocess  # noqa: E402
from train import Sample, augment, export_onnx, load_samples, predict, read_rgb, score, stack  # noqa: E402


def writer(path: Path) -> str:
    return path.name.split("-")[1]


def strong(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """A heavier distortion than `augment`: a smooth random warp of the strokes
    (a coarse displacement grid, like a different hand), then the usual noise.
    Nothing is cut off: a digit lost at the edge would contradict the label."""
    h, w = rgb.shape[:2]
    grid = np.random.default_rng(rng.getrandbits(32)).normal(0, 1, (2, 3, max(3, w // 24)))
    amplitude = rng.uniform(1.0, 2.5)
    dx = cv2.resize(grid[0].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * amplitude
    dy = cv2.resize(grid[1].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * amplitude
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(rgb, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return augment(warped, rng)


def batch(samples: list[Sample], labels: list[str], rng: random.Random, distort) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = stack([preprocess(distort(read_rgb(s.path), rng)) for s in samples])
    targets = [encode(t) for t in labels]
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return images, flat, lengths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--unlabeled", nargs="*", default=[], help="folders of exported cell crops (*.png)")
    parser.add_argument("--kind", default="number", choices=["number", "score"], help="field of --unlabeled crops")
    parser.add_argument("--cvl", type=Path, help="CVL digit strings: adapt on half the writers, score the other half")
    parser.add_argument("--exclude-writers-from", type=Path, nargs="*", default=[])
    parser.add_argument("--synthetic", nargs="*", default=[], help="labelled synthetic shards mixed in")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.9, help="pseudo-labels kept from this confidence")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--oracle", action="store_true", help="train on the true CVL labels (checks the training loop)")
    parser.add_argument("--no-strong", action="store_true", help="only the weak view")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--name", default="digit-crnn-self")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, weights_only=True, map_location=device)
    width = checkpoint.get("width", 1.0)
    model = DigitCRNN(width=width).to(device)
    model.load_state_dict(checkpoint["model_state"])

    unlabeled: list[Sample] = [
        Sample(path, args.kind, "", False, False) for folder in args.unlabeled for path in sorted(Path(folder).rglob("*.png"))
    ]
    held_out: list[Sample] = []
    if args.cvl:
        excluded = {writer(p) for folder in args.exclude_writers_from for p in folder.rglob("*.png")}
        unseen = [p for p in sorted(args.cvl.glob("*.png")) if writer(p) not in excluded]
        # Writers, not strings, are split: the scored half is new handwriting.
        adapt = lambda p: int(hashlib.sha1(writer(p).encode()).hexdigest(), 16) % 2 == 0  # noqa: E731
        unlabeled += [Sample(p, "number", p.name.split("-")[0], False, False) for p in unseen if adapt(p)]
        held_out = [Sample(p, "number", p.name.split("-")[0], False, False) for p in unseen if not adapt(p)]
    if not unlabeled:
        raise SystemExit("nothing to adapt on: give --unlabeled or --cvl")
    synthetic = [s for s in load_samples(sorted({Path(p) for pattern in args.synthetic for p in glob.glob(pattern)}))] if args.synthetic else []
    print(f"device {device}  unlabeled {len(unlabeled)}  held-out {len(held_out)}  synthetic pool {len(synthetic)}", flush=True)

    def report(cycle: int) -> dict:
        row: dict = {"cycle": cycle}
        if held_out:
            numbers = score(held_out, predict(model, held_out, batch_size=1))["number"]
            row.update(held_out_exact=numbers["exact_match"], held_out_cer=numbers["character_error_rate"])
        print(json.dumps(row), flush=True)
        return row

    history = [report(0)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    for cycle in range(1, args.cycles + 1):
        readings = predict(model, unlabeled, batch_size=1)
        kept = [(s, text) for s, (text, confidence) in zip(unlabeled, readings) if text and confidence >= args.confidence]
        if args.oracle:
            kept = [(s, s.text) for s in unlabeled]
        # How good the kept labels are, where the truth is known (CVL only).
        known = [(s, t) for s, t in kept if s.text]
        right = sum(s.text == t for s, t in known) / len(known) if known else None
        print(f"cycle {cycle}: kept {len(kept)}/{len(unlabeled)} pseudo-labels" + (f", {right:.3f} right" if right is not None else ""), flush=True)
        mixed = [(s, s.text) for s in rng.sample(synthetic, min(len(synthetic), len(kept)))] if synthetic else []
        model.train()
        # Batch-norm statistics stay those of the full training: batches here
        # mix wide strings with narrow cells (much white padding), which would
        # shift them and hurt every reading.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        pairs = kept + mixed
        rng.shuffle(pairs)
        for start in range(0, len(pairs), args.batch_size):
            part = pairs[start : start + args.batch_size]
            samples, labels = [s for s, _ in part], [t for _, t in part]
            loss = 0.0
            for distort in (augment,) if args.no_strong else (augment, strong):  # the same label for both views
                images, targets, lengths = (x.to(device) for x in batch(samples, labels, rng, distort))
                log_probs = model(images).log_softmax(dim=2).permute(1, 0, 2)
                frames = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                loss = loss + ctc(log_probs, targets, frames, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        row = report(cycle)
        row.update(kept=len(kept), kept_right=right)
        history.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "width": width, "adapted_from": args.model.name}, args.out / f"{args.name}.pt")
    export_onnx(model.cpu(), args.out / f"{args.name}.onnx")
    (args.out / f"{args.name}.self.json").write_text(
        json.dumps({"from": args.model.name, "confidence": args.confidence, "history": history}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
