# Digit cell recognizer

A small on-device model that reads one handwritten field crop: a score cell, the `Not:` total or the
student number. Names are not read; the student is identified by number.

| | |
|---|---|
| Architecture | fully convolutional CRNN + CTC (no recurrent layers), ~630k parameters, ~2.5 MB ONNX |
| Input | RGB crop → bilinear resize to height 32, width scaled (16–512, multiple of 4), values in [-1, 1] |
| Output | per-frame probabilities over blank + `0123456789.~` |
| Classes | `.` is any decimal separator (written `,` or `.`); `~` means a crossed-out correction is present |
| Confidence | minimum over frames of the winning probability |
| Runtime | `exam_infrastructure::recognition::CellRecognizer` (Rust, `tract-onnx`, no native libraries) |

The preprocessing and decoding in `model.py` are the contract re-implemented in Rust;
`crates/exam_infrastructure/tests/recognizer_parity.rs` checks the Rust reader against fixtures the
Python reference produced.

## Train

Generate data with `tools/synthetic` first: training crops from MNIST's train writers, evaluation
crops from MNIST's test writers.

```powershell
python tools/recognizer/train.py --train "data/synthetic/train/shard-*" `
  --validation data/synthetic/eval/shard-1 --test data/synthetic/eval/shard-2 `
  --out models --name digit-crnn-v1
python tools/recognizer/export_fixtures.py --model models/digit-crnn-v1.onnx `
  --data data/synthetic/eval/shard-2 --out crates/exam_infrastructure/tests/fixtures/recognizer
```

The validation split chooses the best epoch and the confidence threshold for automatic acceptance
(the lowest threshold whose accepted fields reach `--target-accuracy`, 99% by default, with at least
`--min-accepted` fields). The test split is scored once at the end with that threshold. Results,
data provenance and the ONNX hash are written to `models/<name>.metrics.json`.

## Current model: `digit-crnn-v1` (2026-10-10)

Trained 20 epochs (AdamW, one-cycle, max LR 1e-3) on 19,200 synthetic crops (3,200 sheets, MNIST
train writers); validation and test are 1,200 crops each (MNIST test writers). Full numbers:
`models/digit-crnn-v1.metrics.json`. Test split, field-aware decoding:

| Field | Exact match | Character error rate |
|---|---|---|
| Score cells | 96.0% | 2.0% |
| `Not:` total | 89.0% | 4.9% |
| Student number (11 digits) | 64.5% | 4.9% |
| All | 89.6% | 3.9% |

- Crossed-out corrections flagged (`~`): 90.4% recall, 0.09% false alarms; blank cells: 100%.
- Threshold chosen on validation for 99% accuracy: 0.99 → on test, 31.8% of fields are accepted
  automatically with 99.2% accuracy; the rest go to instructor review. Lower thresholds trade
  accuracy for coverage (0.9 → 54.9% accepted at 98.3%).
- Student-number errors are mostly one misread digit (9↔4, 7↔1, 3↔8), which is expected for
  11 digits at ~95% per digit. Matching the number against the course roster (unique candidate at
  edit distance ≤ 1) is the intended fix, not a bigger model.
- A first run with max LR 2e-3 stalled on the CTC "all blank" plateau for 9 of 12 epochs; the
  lower LR left it after 5 epochs.

## Tests

```powershell
python -m unittest discover -s tools/recognizer -p "test_*.py" -v
cargo test -p exam-infrastructure
```

## Limits

All numbers come from synthetic sheets with MNIST digits. They show the pipeline works; they are not
an accuracy claim for real İnönü sheets. Real use needs instructor corrections from the review
screen (stored as crop + corrected value) to fine-tune and to measure real accuracy.
