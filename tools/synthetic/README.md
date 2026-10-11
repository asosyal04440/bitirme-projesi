# Synthetic exam sheets

The project has no real filled exam sheets, so training and benchmark data are generated from the
blank İnönü sheet (`templates/inonu-final-a.docx`). Every sample has exact ground truth because the
generator wrote it.

The published sample is the real exam file with the instructor's name, the question texts and the
document metadata removed; the template image is blanked the same way. Only the header and the
`Soru-PÇ` table are used for reading, so nothing else depends on the removed parts.

Each sample is the blank sheet with:

- a fake student name and 11-digit number in the student's pen (blue/black);
- one score per `Soru-PÇ İlişkisi` column and the `Not:` total in the instructor's pen (often red);
- realistic noise: half points, decimal commas (`7,5`), crossed-out and rewritten scores, blank
  cells, a total that does not match the scores, and text overflowing the cell;
- a simulated phone photo: perspective, rotation, desk background, uneven light, shadows, white
  balance drift, blur, sensor noise and JPEG compression.

All names and numbers are fabricated. Handwriting comes from handwriting-style fonts already on
the machine (Ink Free, Segoe Print/Script, Comic Sans, …), with per-letter jitter, slant, elastic
distortion and pen-width changes. Fonts are read from the system and never copied into outputs.
With `--mnist-dir data/raw/mnist`, digits are real handwritten MNIST digits instead (much closer to
real sheets; recommended). MNIST's train and test splits come from different writers: generate
training data with `--mnist-split train` (default) and evaluation data with `--mnist-split test`,
so a model is never evaluated on handwriting it was trained on.

MNIST (Y. LeCun, C. Cortes, C. J. C. Burges; CC BY-SA 3.0) was downloaded on 2026-10-10 from
`https://storage.googleapis.com/cvdf-datasets/mnist/` into `data/raw/mnist/` and verified against the
official MD5 checksums:

| File | MD5 |
|---|---|
| `train-images-idx3-ubyte.gz` | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | `ec29112dd5afa0611ce80d1b7f02629c` |

## Run

Requires Python 3.11+ with `numpy`, `opencv-python`, `Pillow` (and `PyMuPDF` only for template
extraction): `pip install -r tools/synthetic/requirements.txt`.

```powershell
# training set: real MNIST training-split digits
python tools/synthetic/generate.py --count 5000 --seed 1 --out data/synthetic/train --mnist-dir data/raw/mnist
# evaluation set: different seed and MNIST test-split writers
python tools/synthetic/generate.py --count 500 --seed 2 --out data/synthetic/eval --mnist-dir data/raw/mnist --mnist-split test
```

Options: `--max-scores S1=20,S2=30,S3=25,S4=25` (the sample file only states S1 = 20 and S2 = 30;
S3/S4 default to 25), `--long-side 1600` (photo height in px), `--clean` (no photo effects, for
debugging), `--no-crops`, `--font-dir`, `--mnist-dir`. The same seed always gives the same output.

Output:

| Path | Content |
|---|---|
| `images/<id>.jpg` | the synthetic photo |
| `labels.jsonl` | full truth per sample: values, written text, crossed-out value, field quads, homography, photo parameters |
| `crops/<id>_<field>.png`, `crops/labels.csv` | rectified 64 px-high field crops (with slight detector-like jitter) for training the cell recognizer |
| `golden.json` | benchmark golden file (`tools/benchmark`, schema v1, `corpus_type: synthetic`) |
| `manifest.json` | generator version, seed, template hash, fonts, probabilities |

Field keys follow the benchmark/domain conventions: `student.name`, `student.number`,
`question:1.earned_score` (question ids normalized like the Rust `QuestionId`), and `declared_total`.
Review-required fields mark what the application must send to a human: crossed-out or blank
scores (`question.earned_score`), and a missing or mismatching total (`declared_total`).

Generated data is large and reproducible from the seed; keep it out of version control.

## Another exam sheet

The template (`templates/inonu-final-a.{json,png}`) is extracted from the sheet's vector lines and
printed labels, so the same form with other questions or PÇ pairs needs no code change:

```powershell
soffice --headless --convert-to pdf --outdir build exam.docx
python tools/synthetic/extract_layout.py build/exam.pdf --template-id inonu-final-a
```

## Tests

```powershell
python -m unittest discover -s tools/synthetic -p "test_*.py" -v
```

The tests check that `golden.json` passes the benchmark validator and scores 100% against itself,
that the truth is internally consistent (totals, corrections, blanks, review fields), that projected
field boxes land on the handwriting, and that a seed is reproducible.

## Limits

Synthetic data trains and smoke-tests the pipeline; it cannot prove accuracy on real sheets.
MNIST digits are real handwriting but isolated, US-style and centred; Turkish instructors' digits
(e.g. a crossed 7, a decimal comma) differ, so instructor corrections from the review screen
should complement them later.
