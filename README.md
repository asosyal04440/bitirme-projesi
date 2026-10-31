# Bitirme Projesi — Exam Document Foundation

An offline-first Rust core that turns a phone photo of an İnönü exam sheet into per-student
learning-outcome (PÇ) results, with every uncertain field routed to instructor review. The same
core is meant to run on iOS and Android (Flutter UI planned).

## Workspace layout

- `crates/exam_core`: pure domain — value objects, validation, session aggregation, PÇ results.
- `crates/exam_application`: use cases (`ProcessExamPage`, `FinalizeExamSession`), analyzer and
  repository ports, recognition confidence policy, course roster matching, instrumentation.
- `crates/exam_infrastructure`:
  - `recognition`: on-device digit cell recognizer (ONNX via `tract`, no native libraries);
  - `vision`: page registration against the blank template, cell alignment and cropping (pure Rust);
  - `sheet`: `TemplateSheetAnalyzer`, the `ExamDocumentAnalyzer` for the İnönü sheet;
  - `persistence`: in-memory repository for tests and local development.
- `crates/exam_import`: reads the exam definition from the instructor's Word file — course, term,
  exam type, the `Soru-PÇ` table, maximum scores from the question texts (`(20puan)`,
  `(3*10=30 puan)`) and the table column widths. Missing values are reported, never guessed.
- `crates/exam_cli`: `exam-cli import <docx>` shows what was imported; `exam-cli evaluate` runs the
  full pipeline over synthetic sheets and writes benchmark predictions.

Supporting tools (Python):

- `tools/benchmark`: model-independent evaluator for extraction results against golden data.
- `tools/synthetic`: generator of synthetic filled İnönü sheets (photos, field crops, golden data),
  because no real filled sheets are available.
- `tools/recognizer`: training (GPU), ONNX export and real-handwriting evaluation of the digit
  recognizer (`models/digit-crnn-v4.onnx`), and its comparison with ready-made OCR models.

## How a sheet is read

1. The exam structure (questions, their PÇ, maximum scores, score-cell columns) is imported from the
   exam's Word file; the rest of the header layout is the same on every sheet of the form.
2. The photo is registered against the top of the form (correlation of the printed structure: the
   header, the rules and the Soru-PÇ table's label column; the question columns and question texts
   differ per exam and are not used), then each cell edge snaps to its printed line. The search covers
   the sheet in any of four orientations, partly outside the frame (a close-up of the table) and on
   dark, coloured or textured desks (wide areas that are not white paper are ignored); it runs on all
   CPU cores, about 0.2 s per camera frame on a laptop, and a moving sheet is then tracked.
3. Score cells, the `Not:` total and the student number are read by the 2.5 MB recognizer; names are
   never read — the student number is matched against the course roster.
4. The application validates everything (score ≤ maximum, total = sum of scores, single-PÇ weight
   rule, roster near-matches) and sends anything uncertain to review.

On real handwriting — the CVL Digit Strings evaluation set, 4,687 numbers written by 181 people
whose writing was never used for training — the recognizer reads 83.0% of the 5–7 digit numbers
exactly, with a character error rate of 3.2% (the first, MNIST-trained model: 24.9%). On the same
test, ready-made models do much worse: PaddleOCR's mobile recogniser 39.1%, Microsoft's TrOCR
(handwritten) 1.0%. See `tools/recognizer/README.md`.

End-to-end on 200 synthetic photos with the reader built from the Word file (MNIST digits from
writers never used in training, roster of 193): scores 90.5% exact, student identity 85.8%, 99.0% of
the fields that need review are flagged, cell location IoU 0.92, about 0.3 s per sheet on a laptop
CPU. Matching the whole page instead of the header scores 94.5% here, but only because synthetic
sheets reuse the template's question text; that would not carry over to other exams. These are
synthetic-data results, not an accuracy claim for real sheets.

```text
cargo build --release -p exam-cli
target/release/exam-cli import tools/synthetic/templates/inonu-final-a.docx
target/release/exam-cli evaluate --template tools/synthetic/templates/inonu-final-a.json \
    --model models/digit-crnn-v4.onnx --exam tools/synthetic/templates/inonu-final-a.docx --max-scores S3=25,S4=25 \
    --roster-from-labels --out data/synthetic/e2e/report data/synthetic/e2e/shard-*
python tools/benchmark/evaluate.py --golden data/synthetic/e2e/report/golden.json \
    --predictions data/synthetic/e2e/report/predictions.json
```

## Mobile app (`mobile/`)

Flutter app ("Sınav Okuyucu", Android and iOS) using `shadcn_ui`
components. The Rust core is linked through `flutter_rust_bridge` 2.13 (`mobile/rust`): the
instructor adds an exam from its Word file (missing maximum scores are asked for, a roster CSV is
optional), scans sheets with the camera or gallery, checks or corrects every recognised cell next to
its photo crop and confirms the sheet (the Rust core re-validates the confirmed values: score ≤
maximum, total = sum), and sees PÇ results per sheet and for the class. Exams, sheets and crops are
kept in SQLite; corrected cells are also stored with their crops as training data for the
recognizer. "Excel" exports grades, PÇ and question analysis (`rust_xlsxwriter`) and shares the
file. Everything runs on the device. "Örnek sınavla dene" uses a bundled synthetic sample, so the
app can be shown without paper.

**No set-up: the exam is recognised from the sheet.** "Kâğıt okut" on the home screen opens the
camera; the first sheet shown is enough to set up the exam, without its Word file. The form is a
university standard (the header and the rows of the Soru-PÇ table are always in the same place), so
once the sheet is located the question columns are found from the table's printed lines (any number
of questions, up to 20: the table ends where its horizontal lines do, the columns are the vertical
lines when they come out about equally wide, otherwise an equally spaced layout snapped to the lines;
robust to shadows, blur and a slightly turned photo; tested on 2, 4, 6 and 10 questions,
`tools/synthetic/column_variants.py`), and the printed texts — `(S1)-PÇ2` per column, the course code and
name, the exam kind and term in the title, the group — are read with PaddleOCR's PP-OCRv4 mobile
recognition model (`models/ppocr-v4-rec.onnx`, Apache-2.0, run with `tract` in Rust; 7 crops read in
parallel, about 0.2 s per frame on a laptop). Fields are voted over consecutive frames and the exam is
taken once each field agrees in two frames. The course is created from its code; the instructor only
checks what was read and enters the maximum scores, which are not printed on the sheet. A sheet of an
exam that is already saved goes straight to scanning. The Word import and the roster remain optional.

**Live scanning.** "Kâğıt tara" keeps the camera open: the instructor shows one sheet after
another, the Soru-PÇ table is outlined on the preview, and each sheet is saved with a green tick
(amber when a value needs checking) as soon as it is read, without taking photos. The sheet found
in one frame is tracked in the next (about 20 ms instead of a 180 ms full search), the readings of
consecutive frames are combined (agreeing frames raise the confidence), and a sheet stays saved
while it is in view; the next one is recognised by its different student number. The Excel report
is rewritten after every saved sheet. On the Android emulator a tracked frame takes about 140 ms
and a sheet is saved 0.2–0.9 s after it comes into view. Holding the phone sideways over the top
of the sheet (header and table only) also works.

```text
cd mobile
flutter_rust_bridge_codegen generate    # after changing mobile/rust/src/api
flutter test                            # Dart unit tests
flutter test integration_test -d <device>   # full pipeline on a phone or emulator
flutter build apk --release
```

iOS is built without a Mac by the GitHub Actions workflow `.github/workflows/ios.yml` on a macOS
runner: it builds an unsigned `.ipa` (attached to the release for version tags) and runs the
integration tests on an iPhone simulator. The unsigned `.ipa` can be installed from Windows with a
sideloading tool that signs it with a free Apple ID (Sideloadly, AltStore); TestFlight or the App
Store needs a paid Apple Developer account and signing secrets in the workflow.

The sample exam (`tools/synthetic/templates/inonu-final-a.docx`) and the template image are the real
İnönü sheet with the instructor's name and the question texts removed.

The template, model and sample files in `mobile/assets/` are copies of
`tools/synthetic/templates/`, `models/` and the sample sheet; keep them in sync.

## Verification

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python -m unittest discover -s tools/benchmark -p "test_*.py"
python -m unittest discover -s tools/synthetic -p "test_*.py"
python -m unittest discover -s tools/recognizer -p "test_*.py"
```
