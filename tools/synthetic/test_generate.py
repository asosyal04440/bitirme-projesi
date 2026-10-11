from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))

from evaluate import evaluate, validate_document  # noqa: E402
from extract_layout import MAPPING_RE, Line, Word, _row_cells  # noqa: E402
from generate import DEFAULT_TEMPLATE, Probabilities, generate  # noqa: E402
from handwriting import DigitBank, FontLibrary, render_text, sample_style  # noqa: E402
from photo import PhotoOptions  # noqa: E402

HAS_FONTS = bool(FontLibrary().covering("0123456789.,"))
SMALL = PhotoOptions(long_side=800)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _perfect_predictions(golden: dict) -> dict:
    sessions = copy.deepcopy(golden["sessions"])
    for session in sessions:
        for page in session["pages"]:
            review = page.pop("review_required_fields")
            page.update(
                parse_status="ok",
                parse_error=None,
                confidence_by_field={},
                review_issues=[{"field_key": key, "reason": "low_confidence"} for key in review],
            )
    return {
        "schema_version": 1,
        "manifest": {
            "model_name": "ground-truth-echo",
            "model_revision": "test",
            "artifact_sha256": "0" * 64,
            "artifact_bytes": 1,
            "runtime_name": "none",
            "runtime_version": "0",
            "backend": "cpu",
            "thread_count": 1,
            "temperature": 0,
            "seed": "not_applicable",
            "prompt_version": "none",
            "timestamp": "2026-10-10T00:00:00Z",
            "evaluator_version": "1",
        },
        "sessions": sessions,
    }


@unittest.skipUnless(HAS_FONTS, "no handwriting fonts installed")
class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "run"
        # Raised event rates so every branch appears in a small sample.
        cls.probs = Probabilities(blank_score=0.15, corrected_score=0.3, total_mismatch=0.3, blank_total=0.1)
        generate(16, cls.out, seed=11, photo_options=SMALL, probs=cls.probs)
        cls.labels = _read_jsonl(cls.out / "labels.jsonl")
        cls.golden = json.loads((cls.out / "golden.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_golden_follows_the_benchmark_contract_and_scores_perfectly_against_itself(self) -> None:
        golden = validate_document(copy.deepcopy(self.golden), "golden")
        predictions = validate_document(_perfect_predictions(self.golden), "predictions")
        report = evaluate(golden, predictions)
        metrics = report["metrics"]
        self.assertEqual(metrics["whole_page_perfect_extraction"]["rate"], 1.0)
        self.assertEqual(metrics["whole_session_perfect_extraction"]["rate"], 1.0)
        self.assertEqual(metrics["question_to_outcome_mapping_accuracy"]["rate"], 1.0)

    def test_form_structure_matches_the_printed_table(self) -> None:
        page = self.golden["sessions"][0]["pages"][0]
        self.assertEqual(
            [(m["question_id"], m["outcome_id"], m["weight_basis_points"]) for m in page["mappings"]],
            [("S1", "PÇ2", 10000), ("S2", "PÇ1", 10000), ("S3", "PÇ3", 10000), ("S4", "PÇ2", 10000)],
        )
        self.assertEqual([o["id"] for o in page["outcomes"]], ["PÇ2", "PÇ1", "PÇ3"])
        self.assertTrue(all(o["label"] is None for o in page["outcomes"]))
        self.assertEqual([q["maximum_score"] for q in page["questions"]], ["20", "30", "25", "25"])

    def test_ground_truth_is_internally_consistent(self) -> None:
        seen = {"corrected": 0, "blank": 0, "mismatch": 0}
        for record, session in zip(self.labels, self.golden["sessions"]):
            page = session["pages"][0]
            review = set(page["review_required_fields"])
            cells = {cell["field_key"]: cell for cell in record["cells"]}
            scores = [Decimal(c["value"]) for k, c in cells.items() if k.startswith("question:") and c["value"]]
            self.assertEqual(Decimal(record["computed_total"]), sum(scores, Decimal(0)))
            total = cells["declared_total"]
            if total["value"] is not None:
                self.assertEqual(Decimal(total["value"]) != sum(scores, Decimal(0)), record["total_mismatch"])
                # The handwritten text may use a decimal comma; the value never does.
                self.assertEqual(Decimal(total["written"].replace(",", ".")), Decimal(total["value"]))
            if record["total_mismatch"] or total["value"] is None:
                self.assertIn("declared_total", review)
                seen["mismatch"] += record["total_mismatch"]
            for key, cell in cells.items():
                if not key.startswith("question:"):
                    continue
                if cell["struck"] is not None:
                    seen["corrected"] += 1
                    self.assertNotEqual(cell["struck"], cell["written"])
                    self.assertIn("question.earned_score", review)
                if cell["value"] is None:
                    seen["blank"] += 1
                    self.assertIsNone(cell["written"])
                    self.assertIn("question.earned_score", review)
            for question in page["questions"]:
                if question["earned_score"] is not None:
                    self.assertLessEqual(Decimal(question["earned_score"]), Decimal(question["maximum_score"]))
        self.assertTrue(all(seen.values()), seen)

    def test_regions_and_crops_are_valid(self) -> None:
        with (self.out / "crops" / "labels.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16 * 7)
        for row in rows:
            crop = cv2.imread(str(self.out / row["crop"]))
            self.assertIsNotNone(crop, row["crop"])
            self.assertEqual(crop.shape[0], 64)
        for record in self.labels:
            photo = cv2.imread(str(self.out / record["image"]))
            self.assertEqual(photo.shape[:2], (800, 600))
            for cell in record["cells"]:
                quad = np.array(cell["quad"])
                self.assertTrue((quad >= 0).all() and (quad[:, 0] <= 600).all() and (quad[:, 1] <= 800).all())

    def test_generation_is_deterministic_for_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            generate(3, Path(other), seed=11, photo_options=SMALL, probs=self.probs, crops=False)
            again = _read_jsonl(Path(other) / "labels.jsonl")
            for first, second in zip(self.labels[:3], again):
                self.assertEqual(first, second)
                digest = lambda root, rec: hashlib.sha256((root / rec["image"]).read_bytes()).hexdigest()  # noqa: E731
                self.assertEqual(digest(self.out, first), digest(Path(other), second))

    def test_projected_boxes_land_on_the_handwriting(self) -> None:
        """Without photo noise, written cells contain ink and blank cells do not."""
        with tempfile.TemporaryDirectory() as clean_dir:
            root = Path(clean_dir)
            generate(
                6, root, seed=4, photo_options=PhotoOptions(long_side=1100, degrade=False),
                probs=Probabilities(blank_score=0.3, corrected_score=0, overflow=0),
            )
            checked = {True: 0, False: 0}
            with (root / "crops" / "labels.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if not row["field"].startswith("question:"):
                    continue
                crop = cv2.cvtColor(cv2.imread(str(root / row["crop"])), cv2.COLOR_BGR2RGB).astype(np.int32)
                h, w = crop.shape[:2]
                inner = crop[int(h * 0.2) : int(h * 0.8), int(w * 0.08) : int(w * 0.92)]
                # Ink is coloured or dark; the printed sheet is white with black borders cut away.
                ink = (inner.max(axis=2) - inner.min(axis=2) > 60) | (inner.max(axis=2) < 120)
                written = row["blank"] == "0"
                checked[written] += 1
                if written:
                    self.assertGreater(ink.mean(), 0.01, row["crop"])
                else:
                    self.assertLess(ink.mean(), 0.002, row["crop"])
            self.assertTrue(checked[True] and checked[False], checked)


@unittest.skipUnless(HAS_FONTS, "no handwriting fonts installed")
class HandwritingTests(unittest.TestCase):
    def test_turkish_names_only_use_fonts_with_the_glyphs(self) -> None:
        library = FontLibrary()
        for path in library.covering("Çağrı Işıkoğlu Şimşek"):
            for char in "ÇğıIşŞ":
                self.assertTrue(library.supports(path, char))

    def test_mnist_digits_are_used_when_available(self) -> None:
        rng = np.random.default_rng(0)
        images = np.zeros((10, 28, 28), np.uint8)
        for digit in range(10):
            images[digit, 4:24, 10 + digit // 2 : 14 + digit // 2] = 255
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = struct.pack(">IIII", 0x0803, 10, 28, 28)
            (root / "train-images-idx3-ubyte.gz").write_bytes(gzip.compress(header + images.tobytes()))
            (root / "train-labels-idx1-ubyte").write_bytes(struct.pack(">II", 0x0801, 10) + bytes(range(10)))
            bank = DigitBank(root)
            self.assertEqual(bank.sample(7, rng).shape, (20, 4))
            with self.assertRaises(FileNotFoundError):
                DigitBank(root, "test")  # only the train split exists here
            style = sample_style(FontLibrary().covering("0123456789"), ["black"], rng)
            mask = render_text("12", style, 40, rng, bank)
            self.assertGreater(mask.max(), 0.5)


class LayoutTests(unittest.TestCase):
    def test_row_cells_follow_the_vertical_lines_of_that_row(self) -> None:
        # Geometry of the İnönü score table, in PDF points.
        horizontal = [Line(y, 35.5, 326.4) for y in (195.0, 211.0, 227.0, 243.0)]
        vertical = [Line(35.7, 194.8, 243.3), Line(102.5, 195.3, 242.8), Line(326.1, 194.8, 243.3)]
        vertical += [Line(x, 195.3, 227.3) for x in (158.5, 214.6, 270.7)]
        puan = _row_cells(Word(41.2, 214.8, 61.6, 223.6, "Puan:"), horizontal, vertical)
        self.assertEqual([(c[0], c[2]) for c in puan], [(102.5, 158.5), (158.5, 214.6), (214.6, 270.7), (270.7, 326.1)])
        total = _row_cells(Word(41.2, 230.8, 56.3, 239.6, "Not:"), horizontal, vertical)
        self.assertEqual([(c[0], c[2]) for c in total], [(102.5, 326.1)])

    def test_mapping_cells_parse_question_and_outcome(self) -> None:
        for text, expected in [("(S1)-PÇ2", ("S1", "PÇ2")), ("(S12)-PÇ10", ("S12", "PÇ10")), ("S3 - PÇ3", ("S3", "PÇ3"))]:
            self.assertEqual(MAPPING_RE.match(text).groups(), expected)

    def test_bundled_template_matches_the_sample_sheet(self) -> None:
        template = json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            [(q["id"], q["outcome_id"]) for q in template["questions"]],
            [("S1", "PÇ2"), ("S2", "PÇ1"), ("S3", "PÇ3"), ("S4", "PÇ2")],
        )
        self.assertTrue((DEFAULT_TEMPLATE.parent / template["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
