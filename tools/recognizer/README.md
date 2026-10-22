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
  --out models --name digit-crnn-v4 --epochs 35 --lr 1e-3 --workers 12 --width 1.5
.venv/Scripts/python tools/recognizer/evaluate_real.py --model models/digit-crnn-v4.pt `
  --cvl data/raw/cvl/strings-eval/cvl-strings-eval `
  --exclude-writers-from data/raw/cvl/strings-train data/raw/cvl/cvl-single --out models/digit-crnn-v4.real.json
.venv/Scripts/python tools/recognizer/export_fixtures.py --model models/digit-crnn-v4.onnx `
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
saw**. Strings are decoded like the student-number field (digits only), one image at a time as in
the app (batches pad narrow images with white, which flatters the score by about two points).

| Model | Training digits | Exact match | Character error rate |
|---|---|---|---|
| `digit-crnn-v1` | MNIST + handwriting fonts | 24.9% | 21.9% |
| `digit-crnn-v2` | + EMNIST + USPS | 19.1%\* | 26.7%\* |
| `digit-crnn-v3` | EMNIST + CVL pen digits + 1,262 real strings | 80.7% | 3.7% |
| `digit-crnn-v4` | as v3, 1.5× wider, pen-width augmentation | **83.0%** | **3.2%** |

\* measured with the earlier batched evaluation; the v2 weights were not kept.

v2 shows that more MNIST-style digits do not help: they are thick, centred bitmaps, while pen
writing is thin, slanted and touching. Real pen strokes are what closed the gap.

Caveat: v3 and v4 were trained on other parts of CVL (different writers, but the same forms, pens
and scanning), so CVL measures generalisation to new people more than to new paper and cameras.
Real İnönü sheets remain the decisive test.

### Ready-made models on the same test

`compare_models.py` runs ready-made recognizers on exactly the same strings, digits-only
(`models/comparison.json`):

| Model | Exact match | CER | Time per image | Size |
|---|---|---|---|---|
| **`digit-crnn-v4` (ours)** | **83.0%** | **3.2%** | 1.9 ms (CPU) | 5.4 MB |
| `digit-crnn-v3` (ours) | 80.7% | 3.7% | 1.2 ms (CPU) | 2.4 MB |
| PaddleOCR PP-OCRv4 mobile recognition | 39.1% | 18.7% | 15.1 ms (CPU) | 10.4 MB |
| TrOCR small handwritten (Microsoft) | 1.0% | 106% | 40.7 ms (RTX 4050 GPU) | 235 MB |

PaddleOCR is a general printed-text recogniser (6,600+ characters); handwritten digits are not
what it was trained for. TrOCR was trained on English handwritten text lines and writes words for
digit strings ("References and structures" for 136075); restricted to digit tokens it only repeats
digits. A narrow model trained for the task is both more accurate and far smaller and faster here.

## Current model: `digit-crnn-v4` (2026-10-22)

Width 1.5 (1.42 M parameters, 5.4 MB ONNX), trained 35 epochs on the RTX 4050 (about 66 s per
epoch) on the v3 data: 28,800 synthetic crops (EMNIST and CVL pen digits) and the CVL training
strings four times, with pen-width augmentation. Synthetic test split (MNIST/EMNIST/USPS test
writers): score cells 94.0%, `Not:` total 86.0%, student number 65.0%. On real CVL strings it is
right on 96.5% of the strings it reads with confidence ≥ 0.99 (21% of them) and on 92.7% at ≥ 0.9
(45%). Reading the six cells of a sheet takes about 80 ms on a laptop CPU in Rust (v3: 46 ms);
the extra accuracy is worth it, and confident readings let live scanning save a sheet in fewer
frames. Full numbers: `models/digit-crnn-v4.metrics.json` and `models/digit-crnn-v4.real.json`.

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
