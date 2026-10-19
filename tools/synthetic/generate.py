#!/usr/bin/env python3
"""Generate synthetic filled İnönü exam sheets with exact ground truth.

Each sample is the blank sheet filled with a fake student name and number
(student's pen) and per-question scores plus the `Not:` total (instructor's
pen, often red), then photographed: perspective, lighting, shadow, blur,
noise and JPEG. Outputs:

    images/<id>.jpg          the synthetic photo
    labels.jsonl             full ground truth per sample (values, corrections, field quads)
    crops/<id>_<field>.png   rectified field crops for training a cell recognizer
    crops/labels.csv         crop -> text
    golden.json              benchmark contract (tools/benchmark, schema v1)
    manifest.json            generator settings, for reproducibility

All names and numbers are fabricated; no real student data is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from evaluate import _normalize_question_id  # noqa: E402

from handwriting import DigitBank, DigitSources, FontLibrary, WriterStyle, render_text, sample_style, strike_through  # noqa: E402
from photo import PhotoOptions, composite_ink, photograph, project, rectify  # noqa: E402

GENERATOR_VERSION = "1.0.0"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "inonu-final-a.json"
# The sample sheet states S1 = 20 and S2 = 30 points; S3/S4 are not in the file,
# so they default to 25 each (100 in total). Override with --max-scores.
DEFAULT_MAX_SCORES = {"S1": "20", "S2": "30", "S3": "25", "S4": "25"}
CROP_HEIGHT = 64

FIRST_NAMES = (
    "Ahmet", "Mehmet", "Ayşe", "Fatma", "Emre", "Zeynep", "Elif", "Mustafa", "Can", "Ece",
    "Burak", "Şeyma", "İbrahim", "Gökhan", "Gül", "Özge", "Çağrı", "Büşra", "Oğuz", "Merve",
    "Yusuf", "Selin", "Hüseyin", "Derya", "Kerem", "Irmak", "Tuğba", "Ümit", "Sıla", "Barış",
)
LAST_NAMES = (
    "Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Öztürk", "Aydın", "Özdemir", "Arslan",
    "Doğan", "Kılıç", "Aslan", "Çetin", "Koç", "Kurt", "Özkan", "Şimşek", "Polat", "Güneş",
    "Erdoğan", "Bulut", "Keskin", "Ünal", "Karaca", "Tekin", "Uğurlu", "Işık", "Akgül", "Bozkurt",
)


@dataclass
class Probabilities:
    blank_score: float = 0.03
    half_point: float = 0.15
    comma_decimal: float = 0.5
    corrected_score: float = 0.07
    total_mismatch: float = 0.05
    blank_total: float = 0.03
    missing_name: float = 0.02
    missing_number: float = 0.02
    red_instructor_pen: float = 0.6
    overflow: float = 0.1


@dataclass
class Cell:
    field_key: str
    review_key: str
    written: str | None  # text the writer put down last
    value: str | None  # ground truth value (decimal string for scores)
    struck: str | None = None  # crossed-out earlier value
    extra: dict = field(default_factory=dict)


def score_key(question_id: str) -> str:
    """Benchmark field key of a question's score, using the domain id policy."""
    return f"question:{_normalize_question_id(question_id)}.earned_score"


def _decimal_text(value: Decimal) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _write_decimal(value: Decimal, comma: bool) -> str:
    text = _decimal_text(value)
    return text.replace(".", ",") if comma else text


def _sample_score(maximum: Decimal, rng: np.random.Generator, probs: Probabilities) -> Decimal:
    step = Decimal("0.5") if rng.random() < probs.half_point else Decimal(1)
    steps = int(maximum / step)
    fraction = rng.beta(2.2, 1.4)
    return min(maximum, step * int(round(fraction * steps)))


def sample_sheet(template: dict, max_scores: dict[str, Decimal], rng: np.random.Generator, probs: Probabilities) -> dict:
    comma = bool(rng.random() < probs.comma_decimal)
    name = None if rng.random() < probs.missing_name else f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    number = None if rng.random() < probs.missing_number else "0" + "".join(str(d) for d in rng.integers(0, 10, 10))
    cells = [
        Cell("student.name", "student.name", name, name),
        Cell("student.number", "student.number", number, number),
    ]
    earned: list[Decimal] = []
    for question in template["questions"]:
        qid = question["id"]
        key = score_key(qid)
        if rng.random() < probs.blank_score:
            cells.append(Cell(key, "question.earned_score", None, None, extra={"question": qid}))
            continue
        value = _sample_score(max_scores[qid], rng, probs)
        earned.append(value)
        struck = None
        if rng.random() < probs.corrected_score:
            wrong = value
            while wrong == value:
                wrong = _sample_score(max_scores[qid], rng, probs)
            struck = _write_decimal(wrong, comma)
        cells.append(
            Cell(key, "question.earned_score", _write_decimal(value, comma), _decimal_text(value), struck, {"question": qid})
        )

    total = sum(earned, Decimal(0))
    declared = total
    mismatch = False
    if rng.random() < probs.blank_total:
        cells.append(Cell("declared_total", "declared_total", None, None))
    else:
        if rng.random() < probs.total_mismatch:
            delta = Decimal(int(rng.integers(1, 6))) * (1 if rng.random() < 0.5 or total < 5 else -1)
            declared, mismatch = total + delta, True
        cells.append(Cell("declared_total", "declared_total", _write_decimal(declared, comma), _decimal_text(declared)))
    return {
        "name": name,
        "number": number,
        "cells": cells,
        "computed_total": _decimal_text(total),
        "total_mismatch": mismatch,
    }


def _box_px(box: dict, page_w: int, page_h: int) -> tuple[int, int, int, int]:
    return (
        int(round(box["x"] * page_w)),
        int(round(box["y"] * page_h)),
        int(round(box["width"] * page_w)),
        int(round(box["height"] * page_h)),
    )


def _side_by_side(left: np.ndarray, right: np.ndarray, gap: int) -> np.ndarray:
    rows = max(left.shape[0], right.shape[0])

    def centered(mask: np.ndarray) -> np.ndarray:
        top = (rows - mask.shape[0]) // 2
        return np.pad(mask, ((top, rows - mask.shape[0] - top), (0, 0)))

    return np.hstack([centered(left), np.zeros((rows, gap), np.float32), centered(right)])


def _place(page: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], style: WriterStyle, rng: np.random.Generator, overflow: bool) -> None:
    x, y, w, h = box
    limit_w = w * (1.12 if overflow else 0.9)
    limit_h = h * (1.25 if overflow else 0.85)
    factor = min(1.0, limit_w / mask.shape[1], limit_h / mask.shape[0])
    if factor < 1:
        mask = cv2.resize(mask, (max(1, int(mask.shape[1] * factor)), max(1, int(mask.shape[0] * factor))), interpolation=cv2.INTER_AREA)
    free_x = w - mask.shape[1]
    left = x + int(free_x * np.clip(rng.normal(0.4, 0.2), 0, 1)) if free_x > 0 else x + free_x // 2
    top = y + (h - mask.shape[0]) // 2 + int(rng.normal(0, h * 0.08))
    composite_ink(page, mask, style.ink, left, top)


def fill_sheet(
    blank: np.ndarray,
    template: dict,
    sheet: dict,
    fonts: FontLibrary,
    rng: np.random.Generator,
    probs: Probabilities,
    digits: DigitBank | DigitSources | None,
) -> tuple[np.ndarray, dict]:
    page = np.repeat(blank[..., None], 3, axis=2).astype(np.float32) / 255
    page_h, page_w = blank.shape
    name_fonts = fonts.covering(sheet["name"] or "") if sheet["name"] else fonts.paths
    student = sample_style(name_fonts, ["blue", "dark_blue", "black"], rng)
    student_digits = sample_style(fonts.covering("0123456789"), ["blue", "dark_blue", "black"], rng)
    student_digits = replace(student_digits, ink=student.ink)
    instructor_inks = ["red"] if rng.random() < probs.red_instructor_pen else ["blue", "black"]
    instructor = sample_style(fonts.covering("0123456789.,"), instructor_inks, rng)

    boxes = {key: _box_px(box, page_w, page_h) for key, box in template["fields"].items()}
    for question in template["questions"]:
        qid = question["id"]
        boxes[score_key(qid)] = _box_px(question["score_cell"], page_w, page_h)

    styles = {"student.name": student, "student.number": student_digits}
    for cell in sheet["cells"]:
        if cell.written is None:
            continue
        style = styles.get(cell.field_key, instructor)
        box = boxes[cell.field_key]
        height = int(box[3] * rng.uniform(0.5, 0.8))
        use_digits = digits if cell.field_key != "student.name" else None
        mask = render_text(cell.written, style, height, rng, use_digits)
        if cell.struck is not None:
            struck = strike_through(render_text(cell.struck, style, height, rng, use_digits), rng)
            mask = _side_by_side(struck, mask, gap=int(height * rng.uniform(0.3, 0.7)))
        _place(page, mask, box, style, rng, overflow=bool(rng.random() < probs.overflow))
    return page, boxes


def _quad(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def _normalized_bbox(quad: np.ndarray, width: int, height: int) -> dict[str, float]:
    x0, y0 = np.clip(quad.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(quad.max(axis=0), 0, [width, height])
    return {
        "x": round(float(x0 / width), 6),
        "y": round(float(y0 / height), 6),
        "width": round(float((x1 - x0) / width), 6),
        "height": round(float((y1 - y0) / height), 6),
    }


def _golden_page(template: dict, max_scores: dict[str, Decimal], sheet: dict, regions: dict) -> dict:
    by_key = {cell.field_key: cell for cell in sheet["cells"]}
    questions = []
    for question in template["questions"]:
        qid = question["id"]
        cell = by_key[score_key(qid)]
        questions.append({"id": qid, "earned_score": cell.value, "maximum_score": _decimal_text(max_scores[qid])})
    outcome_ids = list(dict.fromkeys(q["outcome_id"] for q in template["questions"]))
    review = set()
    for cell in sheet["cells"]:
        if cell.value is None or cell.struck is not None:
            review.add(cell.review_key)
    if sheet["total_mismatch"]:
        review.add("declared_total")
    return {
        "page_index": template["page_index"],
        "student": {"name": sheet["name"], "number": sheet["number"]},
        "questions": questions,
        "outcomes": [{"id": outcome_id, "label": None} for outcome_id in outcome_ids],
        "mappings": [
            {"question_id": q["id"], "outcome_id": q["outcome_id"], "weight_basis_points": 10_000}
            for q in template["questions"]
        ],
        "regions": [
            {"field_key": key, "page_index": template["page_index"], "bbox": bbox}
            for key, bbox in regions.items()
        ],
        "review_required_fields": sorted(review),
    }


def generate(
    count: int,
    output: Path,
    seed: int,
    template_path: Path = DEFAULT_TEMPLATE,
    max_scores: dict[str, str] | None = None,
    font_dirs: list[Path] | None = None,
    digits_dir: Path | list[Path] | None = None,
    digits_split: str = "train",
    photo_options: PhotoOptions = PhotoOptions(),
    probs: Probabilities | None = None,
    crops: bool = True,
    images: bool = True,
) -> dict:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    blank = cv2.imread(str(template_path.parent / template["image"]), cv2.IMREAD_GRAYSCALE)
    if blank is None:
        raise FileNotFoundError(template_path.parent / template["image"])
    scores = {k: Decimal(v) for k, v in (max_scores or DEFAULT_MAX_SCORES).items()}
    missing = {q["id"] for q in template["questions"]} - scores.keys()
    if missing:
        raise ValueError(f"no maximum score for questions: {sorted(missing)}")
    fonts = FontLibrary(font_dirs)
    if not fonts.covering("0123456789.,"):
        raise RuntimeError("no handwriting fonts found; pass --font-dir")
    if isinstance(digits_dir, Path):
        digits_dir = [digits_dir]
    if not digits_dir:
        digits = None
    elif len(digits_dir) == 1:
        digits = DigitBank(digits_dir[0], digits_split)
    else:
        digits = DigitSources(digits_dir, digits_split)
    probs = probs or Probabilities()

    output.mkdir(parents=True, exist_ok=True)
    if images:
        (output / "images").mkdir(exist_ok=True)
    if crops:
        (output / "crops").mkdir(parents=True, exist_ok=True)
    sessions, crop_rows = [], []
    seeds = np.random.SeedSequence(seed).spawn(count)
    with (output / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as labels:
        for index in range(count):
            sample_id = f"syn-{seed}-{index:05d}"
            rng = np.random.default_rng(seeds[index])
            sheet = sample_sheet(template, scores, rng, probs)
            page, boxes = fill_sheet(blank, template, sheet, fonts, rng, probs, digits)
            photo, homography, params = photograph(page, rng, photo_options)
            height, width = photo.shape[:2]
            image_name = f"images/{sample_id}.jpg"
            if images:
                cv2.imwrite(str(output / image_name), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
            else:
                image_name = None

            quads = {key: project(homography, _quad(box)) for key, box in boxes.items()}
            regions = {key: _normalized_bbox(quad, width, height) for key, quad in quads.items()}
            if crops:
                for cell in sheet["cells"]:
                    crop_name = f"crops/{sample_id}_{cell.field_key.replace(':', '_').replace('.', '_')}.png"
                    crop = rectify(photo, quads[cell.field_key], CROP_HEIGHT, rng, jitter=0.02 if photo_options.degrade else 0.0)
                    cv2.imwrite(str(output / crop_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    crop_rows.append(
                        {
                            "crop": crop_name,
                            "sample_id": sample_id,
                            "field": cell.field_key,
                            "text": cell.written or "",
                            "value": cell.value or "",
                            "corrected": int(cell.struck is not None),
                            "blank": int(cell.written is None),
                        }
                    )
            record = {
                "sample_id": sample_id,
                "image": image_name,
                "student": {"name": sheet["name"], "number": sheet["number"]},
                "cells": [
                    {
                        "field_key": cell.field_key,
                        "written": cell.written,
                        "value": cell.value,
                        "struck": cell.struck,
                        "quad": [[round(float(v), 2) for v in point] for point in quads[cell.field_key]],
                    }
                    for cell in sheet["cells"]
                ],
                "computed_total": sheet["computed_total"],
                "total_mismatch": sheet["total_mismatch"],
                "homography": [[float(v) for v in row] for row in homography],
                "photo": params,
            }
            labels.write(json.dumps(record, ensure_ascii=False) + "\n")
            sessions.append({"session_id": sample_id, "pages": [_golden_page(template, scores, sheet, regions)]})

    golden = {"schema_version": 1, "corpus_type": "synthetic", "sessions": sessions}
    (output / "golden.json").write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if crops:
        with (output / "crops" / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(crop_rows[0]) if crop_rows else ["crop"])
            writer.writeheader()
            writer.writerows(crop_rows)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "template_id": template["template_id"],
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "seed": seed,
        "count": count,
        "max_scores": {k: _decimal_text(v) for k, v in scores.items()},
        "fonts": [path.name for path in fonts.paths],
        "digit_source": f"{digits.name}-{digits.split}" if digits else "fonts",
        "photo": {"long_side": photo_options.long_side, "degrade": photo_options.degrade},
        "probabilities": probs.__dict__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_scores(text: str) -> dict[str, str]:
    scores = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        Decimal(value)  # validates
        scores[key.strip()] = value.strip()
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--max-scores", type=_parse_scores, help="e.g. S1=20,S2=30,S3=25,S4=25")
    parser.add_argument("--font-dir", type=Path, action="append")
    parser.add_argument(
        "--mnist-dir",
        "--digits-dir",
        type=Path,
        nargs="+",
        help="handwritten digit datasets (MNIST/EMNIST IDX or USPS-style parquet); several are mixed",
    )
    parser.add_argument(
        "--mnist-split",
        choices=["train", "test"],
        default="train",
        help="train for training data, test for evaluation data (different writers)",
    )
    parser.add_argument("--long-side", type=int, default=1600)
    parser.add_argument("--clean", action="store_true", help="no photo degradation (debugging)")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="keep only crops and labels (saves disk)")
    args = parser.parse_args(argv)
    manifest = generate(
        args.count,
        args.out,
        args.seed,
        args.template,
        args.max_scores,
        args.font_dir,
        args.mnist_dir,
        args.mnist_split,
        PhotoOptions(long_side=args.long_side, degrade=not args.clean),
        crops=not args.no_crops,
        images=not args.no_images,
    )
    print(f"generated {manifest['count']} sheets into {args.out} (digits from {manifest['digit_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
