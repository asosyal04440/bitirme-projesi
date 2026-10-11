#!/usr/bin/env python3
"""Extract the field layout of an exam sheet PDF into a reusable template.

The İnönü sheet is a Word document; render it to PDF first, e.g.

    soffice --headless --convert-to pdf --outdir <dir> exam.docx

The table geometry is read from the PDF's vector lines, and cells are located
from their printed labels (`Puan:`, `Not:`, `Soru-PÇ İlişkisi`, `Adı Soyadı`,
`Numarası`), so the same form with a different number of questions or other
question-PÇ pairs is extracted without code changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_VERSION = 1
MAPPING_RE = re.compile(r"^\(?\s*(S\d+)\s*\)?\s*-\s*(\S+)$")


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Line:
    position: float
    start: float
    end: float

    def covers(self, value: float) -> bool:
        return self.start - 0.5 <= value <= self.end + 0.5


@dataclass(frozen=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


Box = tuple[float, float, float, float]


def _lines(page) -> tuple[list[Line], list[Line]]:
    horizontal: set[Line] = set()
    vertical: set[Line] = set()
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.y - b.y) < 0.5:
                    horizontal.add(Line(round(a.y, 1), round(min(a.x, b.x), 1), round(max(a.x, b.x), 1)))
                elif abs(a.x - b.x) < 0.5:
                    vertical.add(Line(round(a.x, 1), round(min(a.y, b.y), 1), round(max(a.y, b.y), 1)))
            elif item[0] == "re":
                rect = item[1]
                if rect.height < 1.5:
                    horizontal.add(Line(round(rect.y0, 1), round(rect.x0, 1), round(rect.x1, 1)))
                elif rect.width < 1.5:
                    vertical.add(Line(round(rect.x0, 1), round(rect.y0, 1), round(rect.y1, 1)))
    by_position = lambda line: line.position  # noqa: E731
    return sorted(horizontal, key=by_position), sorted(vertical, key=by_position)


def _find_word(words: list[Word], prefix: str) -> Word:
    for word in words:
        if word.text.startswith(prefix):
            return word
    raise LayoutError(f"label {prefix!r} was not found on the page")


def _row_bounds(word: Word, horizontal: list[Line]) -> tuple[float, float]:
    x, y = word.center
    above = [line.position for line in horizontal if line.position <= y and line.covers(x)]
    below = [line.position for line in horizontal if line.position >= y and line.covers(x)]
    if not above or not below:
        raise LayoutError(f"no table row encloses label {word.text!r}")
    return max(above), min(below)


def _row_cells(word: Word, horizontal: list[Line], vertical: list[Line]) -> list[Box]:
    """Cells of the row that holds `word`, left to right, starting after its label cell."""
    top, bottom = _row_bounds(word, horizontal)
    middle = (top + bottom) / 2
    xs = sorted({line.position for line in vertical if line.covers(middle)})
    label_x = word.center[0]
    cells = [(left, top, right, bottom) for left, right in zip(xs, xs[1:]) if right - left > 2]
    for index, (left, _, right, _) in enumerate(cells):
        if left <= label_x <= right:
            return cells[index + 1 :]
    raise LayoutError(f"label {word.text!r} is not inside a table cell")


def _text_in(words: list[Word], box: Box) -> str:
    left, top, right, bottom = box
    inside = [w for w in words if left <= w.center[0] <= right and top <= w.center[1] <= bottom]
    return "".join(w.text for w in sorted(inside, key=lambda w: w.x0))


def _normalized(box: Box, width: float, height: float) -> dict[str, float]:
    left, top, right, bottom = box
    return {
        "x": round(left / width, 6),
        "y": round(top / height, 6),
        "width": round((right - left) / width, 6),
        "height": round((bottom - top) / height, 6),
    }


def extract_template(pdf_path: Path, output_dir: Path, template_id: str, dpi: int = 200) -> dict:
    import fitz  # PyMuPDF; only needed for template extraction

    document = fitz.open(pdf_path)
    page = document[0]
    width, height = page.rect.width, page.rect.height
    words = [Word(*w[:5]) for w in page.get_text("words")]
    horizontal, vertical = _lines(page)

    mapping_cells = _row_cells(_find_word(words, "Soru-P"), horizontal, vertical)
    score_cells = _row_cells(_find_word(words, "Puan"), horizontal, vertical)
    total_cells = _row_cells(_find_word(words, "Not"), horizontal, vertical)
    name_cells = _row_cells(_find_word(words, "Soyad"), horizontal, vertical)
    number_cells = _row_cells(_find_word(words, "Numaras"), horizontal, vertical)
    if not total_cells or not name_cells or not number_cells:
        raise LayoutError("student or total cells are missing")

    questions = []
    for mapping_cell in mapping_cells:
        text = _text_in(words, mapping_cell)
        match = MAPPING_RE.match(text)
        if not match:
            raise LayoutError(f"unexpected question-outcome cell text {text!r}")
        center_x = (mapping_cell[0] + mapping_cell[2]) / 2
        score_cell = next((c for c in score_cells if c[0] <= center_x <= c[2]), None)
        if score_cell is None:
            raise LayoutError(f"no score cell under {text!r}")
        questions.append(
            {
                "id": match.group(1),
                "outcome_id": match.group(2),
                "mapping_cell": _normalized(mapping_cell, width, height),
                "score_cell": _normalized(score_cell, width, height),
            }
        )
    if not questions:
        raise LayoutError("no question-outcome pairs were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{template_id}.png"
    page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY).save(output_dir / image_name)
    template = {
        "template_version": TEMPLATE_VERSION,
        "template_id": template_id,
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "page_index": 0,
        "render_dpi": dpi,
        "image": image_name,
        "fields": {
            "student.name": _normalized(name_cells[0], width, height),
            "student.number": _normalized(number_cells[0], width, height),
            "declared_total": _normalized(total_cells[0], width, height),
        },
        "questions": questions,
    }
    (output_dir / f"{template_id}.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "templates")
    parser.add_argument("--template-id", default="inonu-final-a")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        template = extract_template(args.pdf, args.output_dir, args.template_id, args.dpi)
    except LayoutError as exc:
        print(f"layout extraction failed: {exc}", file=sys.stderr)
        return 2
    pairs = ", ".join(f"{q['id']}-{q['outcome_id']}" for q in template["questions"])
    print(f"wrote {args.template_id}: {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
