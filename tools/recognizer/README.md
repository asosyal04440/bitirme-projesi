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

Training data: synthetic İnönü sheets whose digits come from EMNIST (about 3,600 writers) and the
CVL single digits (real pen strokes), plus the 1,262 real CVL training strings. Train on the GPU from
the project `.venv` (Python 3.12 with the CUDA build of PyTorch); `--device auto` picks CUDA.

```powershell
.venv/Scripts/python tools/synthetic/generate.py --count 400 --seed 701 --out data/synthetic/train-v3/shard-1 `
  --mnist-dir data/raw/emnist-digits data/raw/cvl/cvl-single --mnist-split train --no-images
# ... shards 1-12 (seeds 701-712); evaluation shards from MNIST/EMNIST/USPS test writers
.venv/Scripts/python tools/recognizer/train.py --train "data/synthetic/train-v3/shard-*" `
  --real-strings data/raw/cvl/strings-train/train --real-repeat 4 `
  --validation data/synthetic/eval-v2/shard-1 --test data/synthetic/eval-v2/shard-2 `
  --out models --name digit-crnn-v3 --epochs 25 --lr 1e-3 --workers 12
.venv/Scripts/python tools/recognizer/evaluate_real.py --model models/digit-crnn-v3.pt `
  --cvl data/raw/cvl/strings-eval/cvl-strings-eval `
  --exclude-writers-from data/raw/cvl/strings-train data/raw/cvl/cvl-single --out models/digit-crnn-v3.real.json
.venv/Scripts/python tools/recognizer/export_fixtures.py --model models/digit-crnn-v3.onnx `
  --data data/synthetic/eval-v2/shard-2 --out crates/exam_infrastructure/tests/fixtures/recognizer
```

The validation split chooses the best epoch and the confidence threshold for automatic acceptance
(the lowest threshold whose accepted fields reach `--target-accuracy`, 99% by default, with at least
`--min-accepted` fields). The test split is scored once at the end with that threshold. Results,
data provenance and the ONNX hash are written to `models/<name>.metrics.json`.

## Real handwriting: CVL Digit Strings

`evaluate_real.py` scores a model on the CVL Digit Strings evaluation set (ICDAR 2013, TU Wien,
CC BY-NC 4.0): scanned 5–7 digit strings written with pens by real people. Writers that appear in
any CVL training part are excluded, which leaves **4,687 strings by 181 writers the model never
saw**. Strings are decoded like the student-number field (digits only).

| Model | Training digits | Exact match | Character error rate |
|---|---|---|---|
| `digit-crnn-v1` | MNIST + handwriting fonts | 28.6% | 20.2% |
| `digit-crnn-v2` | + EMNIST + USPS | 19.1% | 26.7% |
| `digit-crnn-v3` | EMNIST + CVL pen digits + 1,262 real strings | **83.1%** | **3.2%** |

v2 shows that more MNIST-style digits do not help: they are thick, centred bitmaps, while pen
writing is thin, slanted and touching. Real pen strokes are what closed the gap. The confidence is
meaningful on real writing too: v3 is right on 96.8% of the strings it reads with confidence ≥ 0.99
(15% of them) and on 93.1% at ≥ 0.9 (41%).

## Current model: `digit-crnn-v3` (2026-10-19)

Trained 25 epochs on the RTX 4050 (about 52 s per epoch) on 33,848 crops: 28,800 synthetic ones
(4,800 sheets) and the CVL training strings four times. Synthetic test split (MNIST/EMNIST/USPS test
writers), field-aware decoding: score cells 93.1%, `Not:` total 83.5%, student number 63.0%
(v1: 96.0%, 89.0%, 64.5%). On synthetic sheets v1 is slightly better because its training digits
look like the test digits; on real handwriting v3 is far better, and that is what the app needs.
Full numbers: `models/digit-crnn-v3.metrics.json` and `models/digit-crnn-v3.real.json`.

- Threshold chosen on validation for 99% accuracy: 0.99 → on the synthetic test, 42.8% of fields
  are accepted automatically with 99.2% accuracy (v1: 31.8%); the rest go to instructor review.
- Student-number errors are mostly one misread digit; matching the number against the course roster
  (unique candidate at edit distance ≤ 1) handles them.
- A first run with max LR 2e-3 stalled on the CTC "all blank" plateau; max LR 1e-3 is used since.

## Tests

```powershell
python -m unittest discover -s tools/recognizer -p "test_*.py" -v
cargo test -p exam-infrastructure
```

## Limits

CVL strings are real handwriting but not İnönü sheets: they are clean scans of whole numbers, not
phone photos of table cells. They measure how well the model reads human pen digits; accuracy on
real sheets still has to be confirmed with instructor corrections from the review screen (stored on
the device as crop + corrected value), which can also fine-tune the model.
