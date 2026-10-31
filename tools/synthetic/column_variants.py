"""Sheets whose Soru-PÇ table has another number of questions than the template.

The form is a university standard: the header and the table rows are always in
the same place, only the number of question columns (and their PÇ) depends on
the exam. These sheets test that the exam is still recognised from the photo:
the template's table is redrawn with 2, 6 and 10 columns, then photographed.

    python tools/synthetic/column_variants.py \
        crates/exam_infrastructure/tests/fixtures/detect
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from photo import PhotoOptions, photograph  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "templates" / "inonu-final-a.png"

# Table geometry of the template (1700 x 2200 px): the fixed label column ends
# at LABEL_RIGHT; rows are the mapping, the scores and the total ("Not").
LABEL_RIGHT = 284
ROWS = (541, 586, 630, 675)
COLUMN = 155  # the template's question column width
TABLE_LIMIT = 1594  # right edge of the header table
LINE = 2

# Questions -> (photo seed, PÇ of each question). The seed picks the desk and
# lighting; on a near-white desk with the sheet's edges lost (e.g. seed 102
# for 10 questions) the sheet is not found and the frame is skipped, which
# is not what these sheets are for.
VARIANTS = {
    2: (100, [1, 2]),
    6: (101, [1, 2, 3, 1, 2, 4]),
    10: (200, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
}


def redraw_table(page: Image.Image, outcomes: list[int]) -> Image.Image:
    page = page.convert("RGB").copy()
    draw = ImageDraw.Draw(page)
    count = len(outcomes)
    width = min(COLUMN, (TABLE_LIMIT - LABEL_RIGHT) // count)
    right = LABEL_RIGHT + width * count
    # Clear the template's question columns, keep the label column.
    draw.rectangle([LABEL_RIGHT + LINE, ROWS[0] - 4, 1650, ROWS[-1] + 4], fill="white")
    for y in ROWS:
        draw.rectangle([LABEL_RIGHT, y - 1, right, y], fill="black")
    for k in range(1, count + 1):
        x = LABEL_RIGHT + k * width
        bottom = ROWS[-1] if k == count else ROWS[2]
        draw.rectangle([x - 1, ROWS[0], x, bottom], fill="black")
    bold = ImageFont.truetype(str(HERE / "timesbd.ttf"), 22)
    regular = ImageFont.truetype(str(HERE / "times.ttf"), 22)
    for k, outcome in enumerate(outcomes):
        head, tail = f"(S{k + 1})", f"-PÇ{outcome}"
        size = draw.textlength(head, font=bold) + draw.textlength(tail, font=regular)
        x = LABEL_RIGHT + k * width + (width - size) / 2
        y = (ROWS[0] + ROWS[1]) / 2 - 12
        draw.text((x, y), head, font=bold, fill="black")
        draw.text((x + draw.textlength(head, font=bold), y), tail, font=regular, fill="black")
    return page


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    template = Image.open(TEMPLATE)
    labels = []
    for count, (seed, outcomes) in VARIANTS.items():
        page = redraw_table(template, outcomes)
        flat = np.asarray(page, dtype=np.float32) / 255.0
        rng = np.random.default_rng(seed)
        photo, _, _ = photograph(flat, rng, PhotoOptions(long_side=1600, degrade=True))
        name = f"columns-{count}.jpg"
        Image.fromarray(photo).save(out / name, quality=90)
        labels.append({"image": name, "outcomes": [f"PÇ{o}" for o in outcomes]})
    (out / "columns.jsonl").write_text(
        "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
