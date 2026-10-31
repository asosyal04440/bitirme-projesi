//! Recognising the exam from the sheet itself, without its Word file.
//!
//! The form is a university standard: the header and the rows of the
//! `Soru-PÇ` table are always in the same place, only the texts and the
//! number of question columns change. So once the sheet is located, the
//! question columns are found from the table's printed vertical lines and
//! the printed texts are read with [`PrintedTextReader`]: the PÇ of each
//! question (`(S1)-PÇ2`), the course (`BİLM374-2022-YAPAY ZEKA`), the exam
//! kind and term from the title, and the group. Maximum scores are not on the
//! table (they are in the question texts) and are left to the instructor.

use crate::printed::PrintedTextReader;
use crate::recognition::RecognitionError;
use crate::sheet::{Located, SheetTemplate, TableRows};
use crate::vision::{warp_quad, Point, Rect};
use exam_core::domain::{LearningOutcomeId, QuestionId};
use exam_import::{ColumnSpan, ExamMetadata, ImportWarning, ImportedExam, ImportedQuestion};
use image::RgbImage;

/// Output pixels per template pixel when the table band is sampled.
const BAND_SCALE: f64 = 3.0;
/// A sample is ink when it is this much darker than the paper; printed table
/// lines in a phone photo are grey rather than black.
const DARK: f32 = 0.82;
/// A sample is (faint) ink when it is this much darker than the paper.
const FAINT: f32 = 0.92;
/// Share of the two table rows a column must be dark in to be a printed line
/// (handwritten digits cover far less of it).
const LINE_COVERAGE: f32 = 0.7;
/// Template pixels around a column that the local paper level is taken from.
const PAPER_REACH: f64 = 12.0;
/// Average strength of the inner lines below which the table is taken to have
/// a single question column.
const MIN_SUPPORT: f32 = 0.35;
/// Samples a printed line may drift sideways across the band (a tilted photo).
const TILT: usize = 1;
/// Narrowest question column, in template pixels (a label is 4-5 characters).
const MIN_COLUMN: f64 = 20.0;
/// Template pixels the first column's left line may lie from where the
/// registration puts it.
const START_REACH: f64 = 16.0;
/// Samples a line may lie apart in the table's two rows.
const ROW_DRIFT: usize = 4;
/// Template pixels of the table band over which the rows' offset is found.
const OFFSET_STRETCH: f64 = 24.0;
/// Units per template pixel in the column spans of the detected exam.
const UNITS_PER_PIXEL: f64 = 20.0;

#[derive(Debug)]
pub enum DetectError {
    NotLocated,
    NoTable,
    Text(RecognitionError),
}

impl std::fmt::Display for DetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLocated => write!(f, "kâğıt bulunamadı"),
            Self::NoTable => write!(f, "Soru-PÇ tablosunun sütunları bulunamadı"),
            Self::Text(e) => write!(f, "yazı okunamadı: {e}"),
        }
    }
}

impl std::error::Error for DetectError {}

impl From<RecognitionError> for DetectError {
    fn from(error: RecognitionError) -> Self {
        Self::Text(error)
    }
}

/// Reads the exam definition off a located sheet. The result has the shape of
/// an exam imported from Word, with the column spans in units of
/// 1/20 template pixel and no maximum scores (one warning per question).
pub fn detect_exam(
    template: &SheetTemplate,
    text: &PrintedTextReader,
    photo: &RgbImage,
    located: &Located,
) -> Result<ImportedExam, DetectError> {
    if !located.found() {
        return Err(DetectError::NotLocated);
    }
    let lines = table_lines(template, photo, located).ok_or(DetectError::NoTable)?;
    let rows = template.table_rows();

    // Every printed text to read: the Soru-PÇ labels, then the header.
    let mut rects: Vec<Rect> = lines
        .windows(2)
        .map(|pair| Rect {
            x: pair[0],
            y: rows.mapping.0,
            width: pair[1] - pair[0],
            height: rows.mapping.1,
        })
        .collect();
    let header = template.header().copied();
    if let Some(h) = header {
        // The title is too long to read in one piece without squeezing its
        // letters: its two halves are read separately, overlapping enough that
        // every word in the middle is whole in one of them.
        let half = h.title.width * 0.6;
        let left = Rect {
            width: half,
            ..h.title
        };
        let right = Rect {
            x: h.title.x + h.title.width - half,
            width: half,
            ..h.title
        };
        rects.extend([h.course, left, right]);
        rects.extend(h.group);
    }
    let crops: Vec<RgbImage> = rects
        .iter()
        .map(|&rect| tighten(&cut(template, photo, located, rect)))
        .collect();
    let mut texts = read_all(text, &crops)?.into_iter();

    let mut questions = Vec::new();
    for (index, pair) in lines.windows(2).enumerate() {
        let label = texts.next().unwrap_or_default();
        let (number, outcomes) = parse_mapping(&label).unwrap_or((index as u32 + 1, Vec::new()));
        let printed_id = format!("S{number}");
        let Ok(id) = QuestionId::new(&printed_id) else {
            continue;
        };
        questions.push(ImportedQuestion {
            printed_id,
            id,
            outcomes: outcomes
                .iter()
                .filter_map(|n| LearningOutcomeId::new(format!("PÇ{n}")).ok())
                .collect(),
            maximum_score: None,
            maximum_source: None,
            score_column: span(pair[0], pair[1]),
        });
    }
    if questions.is_empty() {
        return Err(DetectError::NoTable);
    }

    let mut metadata = ExamMetadata::default();
    if let Some(h) = header {
        let course = texts.next().unwrap_or_default();
        let (code, name) = parse_course(&course);
        metadata.course_code = code;
        metadata.course_name = name;
        metadata.course = Some(course).filter(|c| !c.is_empty());
        let left = texts.next().unwrap_or_default();
        let right = texts.next().unwrap_or_default();
        let title = parse_title(&format!("{left} {right}"));
        metadata.exam_kind = title.kind;
        metadata.academic_year = title.year;
        metadata.term = title.term;
        if h.group.is_some() {
            metadata.group = parse_group(&texts.next().unwrap_or_default());
        }
    }
    let warnings = questions
        .iter()
        .map(|q| ImportWarning::MissingMaximumScore(q.id.clone()))
        .collect();
    Ok(ImportedExam {
        metadata,
        questions,
        page_width_twips: (template.size().0 as f64 * UNITS_PER_PIXEL) as u32,
        total_column: span(lines[0], *lines.last().expect("two lines at least")),
        warnings,
    })
}

/// Reads every crop, spread over the CPU's cores (a phone has several; one
/// read takes tens of milliseconds).
fn read_all(text: &PrintedTextReader, crops: &[RgbImage]) -> Result<Vec<String>, DetectError> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, crops.len().max(1));
    let chunk = crops.len().div_ceil(threads).max(1);
    let results: Vec<Result<Vec<String>, RecognitionError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = crops
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|crop| text.read(crop).map(|t| t.text))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("text reading thread"))
            .collect()
    });
    let mut all = Vec::with_capacity(crops.len());
    for part in results {
        all.extend(part?);
    }
    Ok(all)
}

fn span(left: f64, right: f64) -> ColumnSpan {
    ColumnSpan {
        left_twips: (left * UNITS_PER_PIXEL).round() as u32,
        width_twips: ((right - left) * UNITS_PER_PIXEL).round() as u32,
    }
}

/// The photo inside `rect` (template pixels), upright, 64 px high, with a
/// small inset so cell borders are not read as text.
fn cut(template: &SheetTemplate, photo: &RgbImage, located: &Located, rect: Rect) -> RgbImage {
    let inset = Rect {
        x: rect.x + 2.0,
        y: rect.y + 1.5,
        width: (rect.width - 4.0).max(1.0),
        height: (rect.height - 3.0).max(1.0),
    };
    let to_photo = template.to_photo(located).expect("located sheet");
    let quad = inset.corners().map(|p| to_photo.apply(p));
    let height = 64;
    let width = ((inset.width / inset.height) * f64::from(height))
        .round()
        .max(8.0) as u32;
    warp_quad(photo, &quad, width, height)
}

/// The text inside a cell crop, cut to its ink with some margin: the reader
/// expects a line of text that fills the height, while a table cell is much
/// taller than its label. Rows and columns that are dark almost end to end
/// (cell borders) are not ink.
fn tighten(crop: &RgbImage) -> RgbImage {
    let (w, h) = (crop.width() as usize, crop.height() as usize);
    let values: Vec<f32> = crop
        .pixels()
        .map(|p| f32::from(p[0].max(p[1]).max(p[2])))
        .collect();
    let mut sorted = values.clone();
    sorted.sort_by(f32::total_cmp);
    let paper = sorted[(sorted.len() * 9) / 10].max(1.0);
    let ink = |x: usize, y: usize| values[y * w + x] < paper * 0.75;
    // Computed once: asking per pixel would make this cubic in the crop size.
    let border_row: Vec<bool> = (0..h)
        .map(|y| (0..w).filter(|&x| ink(x, y)).count() > w * 7 / 10)
        .collect();
    let border_col: Vec<bool> = (0..w)
        .map(|x| (0..h).filter(|&y| ink(x, y)).count() > h * 7 / 10)
        .collect();
    let text = |x: usize, y: usize| ink(x, y) && !border_row[y] && !border_col[x];
    let rows: Vec<usize> = (0..h).filter(|&y| (0..w).any(|x| text(x, y))).collect();
    let cols: Vec<usize> = (0..w).filter(|&x| (0..h).any(|y| text(x, y))).collect();
    let (Some(&top), Some(&bottom), Some(&left), Some(&right)) =
        (rows.first(), rows.last(), cols.first(), cols.last())
    else {
        return crop.clone();
    };
    let margin = ((bottom - top + 1) as f32 * 0.3).ceil() as usize + 1;
    let (x0, y0) = (left.saturating_sub(margin), top.saturating_sub(margin));
    let (x1, y1) = ((right + margin).min(w - 1), (bottom + margin).min(h - 1));
    image::imageops::crop_imm(
        crop,
        x0 as u32,
        y0 as u32,
        (x1 - x0 + 1) as u32,
        (y1 - y0 + 1) as u32,
    )
    .to_image()
}

/// Template x of the table's vertical lines, left to right, starting at the
/// left edge of the first question column and ending at the table's right
/// edge. `None` when fewer than two lines are found.
pub fn table_lines(
    template: &SheetTemplate,
    photo: &RgbImage,
    located: &Located,
) -> Option<Vec<f64>> {
    let rows = template.table_rows();
    let (width, _) = template.size();
    let left = rows.first_column - 12.0;
    // Within the header table above, which the question table never passes.
    let right = width as f64 * 0.94;
    let to_photo = template.to_photo(located)?;
    // The table's rows with a row's height of margin above and below, sampled
    // in one band: the registration may be a few pixels off, more so towards
    // the right, so the rows are found again from their printed lines.
    let pad = rows.mapping.1;
    let (top, bottom) = (rows.mapping.0 - pad, rows.total.0 + rows.total.1 + pad);
    let band = Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    };
    let quad: [Point; 4] = band.corners().map(|p| to_photo.apply(p));
    let (bw, bh) = (
        (band.width * BAND_SCALE).round().max(1.0) as usize,
        (band.height * BAND_SCALE).round().max(4.0) as usize,
    );
    let image = warp_quad(photo, &quad, bw as u32, bh as u32);
    let values: Vec<f32> = image
        .pixels()
        .map(|p| f32::from(p[0].max(p[1]).max(p[2])))
        .collect();
    // Paper level around each column (shadows fall across the table): the
    // column's bright end, then the brightest of its neighbours.
    let column_paper: Vec<f32> = (0..bw)
        .map(|x| {
            let mut column: Vec<f32> = (0..bh).map(|y| values[y * bw + x]).collect();
            column.sort_by(f32::total_cmp);
            column[(column.len() * 9) / 10]
        })
        .collect();
    let reach = (PAPER_REACH * BAND_SCALE) as usize;
    let paper: Vec<f32> = (0..bw)
        .map(|x| {
            column_paper[x.saturating_sub(reach)..(x + reach + 1).min(bw)]
                .iter()
                .copied()
                .fold(1.0_f32, f32::max)
        })
        .collect();
    let dark = |x: usize, y: usize| values[y * bw + x] < paper[x] * DARK;
    // Lines blurred thin and grey (a close-up of a small print) are only
    // slightly darker than the paper.
    let faint = |x: usize, y: usize| values[y * bw + x] < paper[x] * FAINT;
    let offsets = row_offsets(&rows, top, pad, bw, bh, &dark);
    // Each row is sampled only in its strips above and below the text (the
    // label is centred, the handwriting mostly too): a printed line crosses
    // them, text does not. A line must show in both rows.
    let strip_coverage = |(y, height): (f64, f64)| -> Vec<f32> {
        let strip: Vec<f64> = (0..20)
            .map(|k| (f64::from(k) + 0.5) / 20.0)
            .filter(|t| (0.1..0.3).contains(t) || (0.7..0.9).contains(t))
            .map(|t| (y - top + t * height) * BAND_SCALE)
            .collect();
        (0..bw)
            .map(|x| {
                let (from, to) = (x.saturating_sub(TILT), (x + TILT).min(bw - 1));
                strip
                    .iter()
                    .filter_map(|&r| {
                        let r = (r + offsets[x]).round();
                        (r >= 0.0 && (r as usize) < bh).then_some(r as usize)
                    })
                    .filter(|&r| (from..=to).any(|c| dark(c, r)))
                    .count() as f32
                    / strip.len().max(1) as f32
            })
            .collect()
    };
    let mapping = strip_coverage(rows.mapping);
    let score = strip_coverage(rows.score);
    // A line must show in both rows, but a slightly turned photo puts it a
    // few samples apart in the two.
    let widest = |row: &[f32], x: usize| {
        row[x.saturating_sub(ROW_DRIFT)..(x + ROW_DRIFT + 1).min(row.len())]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    };
    let coverage: Vec<f32> = (0..mapping.len())
        .map(|x| {
            mapping[x]
                .min(widest(&score, x))
                .max(score[x].min(widest(&mapping, x)))
        })
        .collect();
    let to_template = |sample: usize| left + (sample as f64 + 0.5) / BAND_SCALE;
    let to_sample = |x: f64| ((x - left) * BAND_SCALE).round().max(0.0) as usize;
    // Strongest line within `reach` template pixels of `x`.
    let near = |x: f64, reach: f64| -> (f32, f64) {
        let (from, to) = (
            to_sample(x - reach),
            to_sample(x + reach).min(coverage.len() - 1),
        );
        (from..=to)
            .map(|s| (coverage[s], to_template(s)))
            .fold((0.0, x), |best, c| if c.0 > best.0 { c } else { best })
    };
    // The first column starts where the form's fixed label column ends; the
    // table ends at its rightmost printed line.
    let strong: Vec<f64> = peaks(&coverage, LINE_COVERAGE, 4)
        .into_iter()
        .map(to_template)
        .collect();
    let start = strong
        .iter()
        .copied()
        .filter(|x| (x - rows.first_column).abs() <= START_REACH)
        .min_by(|a, b| {
            (a - rows.first_column)
                .abs()
                .total_cmp(&(b - rows.first_column).abs())
        })
        .unwrap_or_else(|| near(rows.first_column, 6.0).1);
    // The table ends where its horizontal lines do (the sheet's edge or a
    // shadow beyond it is a vertical line too, but carries none).
    let lines_at = |x: usize| {
        [
            rows.mapping.0,
            rows.score.0,
            rows.total.0,
            rows.total.0 + rows.total.1,
        ]
        .iter()
        .filter(|&&y| {
            let r = (y - top) * BAND_SCALE + offsets[x];
            (r.round() as i64 - 4..=r.round() as i64 + 4)
                .any(|r| r >= 0 && (r as usize) < bh && faint(x, r as usize))
        })
        .count()
            >= 2
    };
    let first = to_sample(start).min(bw - 1);
    let (mut extent, mut gap) = (first, 0);
    for x in first..bw {
        if lines_at(x) {
            (extent, gap) = (x, 0);
        } else {
            gap += 1;
            if gap as f64 > MIN_COLUMN * BAND_SCALE {
                break;
            }
        }
    }
    let extent = to_template(extent);
    let end = strong
        .iter()
        .copied()
        .rfind(|&x| x > start + MIN_COLUMN && x <= extent + 4.0)
        .or_else(|| strong.iter().copied().rfind(|&x| x > start + MIN_COLUMN))?;
    // The printed lines themselves, when they split the table into columns
    // of about the same width (the form's columns are equally wide; a column
    // may come out somewhat off where the registration is).
    let printed: Vec<f64> = strong
        .iter()
        .copied()
        .filter(|&x| x > start + MIN_COLUMN / 2.0 && x < end - MIN_COLUMN / 2.0)
        .collect();
    let mut lines = vec![start];
    lines.extend(printed);
    lines.push(end);
    let widths: Vec<f64> = lines.windows(2).map(|w| w[1] - w[0]).collect();
    let mut sorted = widths.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    if median >= MIN_COLUMN && widths.iter().all(|w| (w / median - 1.0).abs() <= 0.3) {
        return Some(lines);
    }
    // Otherwise (a line too faint, or handwriting across one): question
    // columns are equally wide in the form's table: try every count,
    // each inner line snapped to the nearest printed line (columns may differ
    // by a few pixels). A table of 4 columns also has the middle line of a
    // 2-column layout, so the most columns that are still well supported win.
    let layouts: Vec<(f32, Vec<f64>)> = (2..=20_usize)
        .map_while(|count| {
            let width = (end - start) / count as f64;
            (width >= MIN_COLUMN).then(|| {
                let inner: Vec<(f32, f64)> = (1..count)
                    .map(|k| near(start + k as f64 * width, width * 0.08))
                    .collect();
                let support = inner.iter().map(|(c, _)| *c).sum::<f32>() / inner.len() as f32;
                let mut lines = vec![start];
                lines.extend(inner.iter().map(|(_, x)| *x));
                lines.push(end);
                (support, lines)
            })
        })
        .collect();
    let strongest = layouts.iter().map(|(s, _)| *s).fold(0.0_f32, f32::max);
    if strongest < MIN_SUPPORT {
        // No inner line anywhere: a single question.
        return Some(vec![start, end]);
    }
    layouts
        .into_iter()
        .filter(|(support, _)| *support >= strongest * 0.8)
        .map(|(_, lines)| lines)
        .next_back()
}

/// For each sample column of the table band, how many samples the table's
/// rows lie lower than the registration put them: found per stretch of the
/// band from the horizontal lines between the rows, then smoothed. Where no
/// lines show (right of the table), the nearest stretch's offset is used.
fn row_offsets(
    rows: &TableRows,
    top: f64,
    pad: f64,
    width: usize,
    height: usize,
    dark: &impl Fn(usize, usize) -> bool,
) -> Vec<f64> {
    let lines: Vec<f64> = [
        rows.mapping.0,
        rows.score.0,
        rows.total.0,
        rows.total.0 + rows.total.1,
    ]
    .iter()
    .map(|y| (y - top) * BAND_SCALE)
    .collect();
    let chunk = (OFFSET_STRETCH * BAND_SCALE) as usize;
    let range = (pad * 0.8 * BAND_SCALE) as i64;
    let found: Vec<Option<f64>> = (0..width.div_ceil(chunk))
        .map(|c| {
            let columns = c * chunk..((c + 1) * chunk).min(width);
            let count = columns.len() as f32;
            let profile: Vec<f32> = (0..height)
                .map(|y| columns.clone().filter(|&x| dark(x, y)).count() as f32 / count)
                .collect();
            let at = |y: i64| {
                (y - 1..=y + 1)
                    .filter(|&r| r >= 0 && (r as usize) < height)
                    .map(|r| profile[r as usize])
                    .fold(0.0_f32, f32::max)
            };
            let (best, shift) = (-range..=range)
                .map(|dy| {
                    let support: f32 = lines.iter().map(|&l| at(l.round() as i64 + dy)).sum();
                    (support, dy)
                })
                .fold((0.0_f32, 0), |a, b| if b.0 > a.0 { b } else { a });
            // Most of the stretch dark on most of the lines.
            (best >= 0.5 * lines.len() as f32).then_some(shift as f64)
        })
        .collect();
    let known: Vec<(usize, f64)> = found
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.map(|v| (i, v)))
        .collect();
    if known.is_empty() {
        return vec![0.0; width];
    }
    let filled: Vec<f64> = (0..found.len())
        .map(|i| {
            known
                .iter()
                .min_by_key(|(k, _)| k.abs_diff(i))
                .map(|&(_, v)| v)
                .unwrap_or(0.0)
        })
        .collect();
    // Median of each stretch and its neighbours: one stretch misled by
    // handwriting or a crossed-out line does not count.
    let smoothed: Vec<f64> = (0..filled.len())
        .map(|i| {
            let mut near: Vec<f64> =
                filled[i.saturating_sub(2)..(i + 3).min(filled.len())].to_vec();
            near.sort_by(f64::total_cmp);
            near[near.len() / 2]
        })
        .collect();
    (0..width).map(|x| smoothed[x / chunk]).collect()
}

/// Centres of the runs where `values` reaches `threshold`; runs closer than
/// `merge` samples are joined.
fn peaks(values: &[f32], threshold: f32, merge: usize) -> Vec<usize> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v < threshold {
            continue;
        }
        match runs.last_mut() {
            Some((_, end)) if i <= *end + merge => *end = i,
            _ => runs.push((i, i)),
        }
    }
    runs.into_iter().map(|(a, b)| (a + b) / 2).collect()
}

/// Upper case without Turkish letters, so the reader's spelling does not
/// matter (its character set has no Ç, Ğ, İ, Ş).
fn fold(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'ç' | 'Ç' => 'C',
            'ğ' | 'Ğ' => 'G',
            'ı' | 'I' | 'i' | 'İ' => 'I',
            'ö' | 'Ö' => 'O',
            'ş' | 'Ş' => 'S',
            'ü' | 'Ü' => 'U',
            'â' | 'Â' => 'A',
            other => other.to_ascii_uppercase(),
        })
        .collect()
}

/// A number as printed, read with the usual confusions (`I`/`l`/`|` for 1,
/// `O` for 0).
fn digits(token: &str) -> Option<u32> {
    let mapped: String = token
        .chars()
        .map(|c| match c {
            'I' | 'L' | '|' | 'l' | 'i' => '1',
            'O' | 'o' => '0',
            other => other,
        })
        .collect();
    mapped.parse().ok()
}

/// `(S1)-PÇ2` → question 1, PÇ 2; several PÇ (`(S3)-PÇ1,PÇ4`) are kept.
pub fn parse_mapping(text: &str) -> Option<(u32, Vec<u32>)> {
    let folded: String = fold(text).chars().filter(|c| !c.is_whitespace()).collect();
    let number_chars = |c: char| c.is_ascii_digit() || matches!(c, 'I' | 'L' | '|' | 'O');
    let s_pos = folded.find('S')?;
    let question: String = folded[s_pos + 1..]
        .chars()
        .take_while(|&c| number_chars(c))
        .collect();
    let question = digits(&question)?;
    let mut outcomes = Vec::new();
    let mut rest = &folded[s_pos + 1..];
    while let Some(p) = rest.find("PC") {
        let number: String = rest[p + 2..]
            .chars()
            .take_while(|&c| number_chars(c))
            .collect();
        if let Some(n) = digits(&number) {
            outcomes.push(n);
        }
        rest = &rest[p + 2..];
    }
    Some((question, outcomes))
}

/// `BİLM374-2022-YAPAY ZEKA` → (`BİLM374`, `YAPAY ZEKA`).
pub fn parse_course(text: &str) -> (Option<String>, Option<String>) {
    let text = text.trim();
    let mut parts = text.split('-').map(str::trim).filter(|p| !p.is_empty());
    let code = parts
        .next()
        .map(|c| c.replace(' ', "").to_uppercase())
        .map(|c| {
            // Course codes are written with a dotted capital I (BİLM): the reader
            // gives "i" or "I".
            c.replace(['I', 'ı'], "İ")
        });
    let name = text
        .rsplit('-')
        .next()
        .map(str::trim)
        .filter(|n| !n.is_empty() && Some(n.replace(' ', "").to_uppercase()) != code)
        .map(|n| n.to_uppercase());
    (
        code.filter(|c| c.chars().any(|ch| ch.is_ascii_digit())),
        name,
    )
}

pub struct Title {
    pub kind: Option<String>,
    pub year: Option<String>,
    pub term: Option<String>,
}

/// `2025 - 2026 EĞİTİM-ÖĞRETİM YILI BAHAR DÖNEMİ FİNAL SINAV KÂĞIDI`.
pub fn parse_title(text: &str) -> Title {
    let folded: String = fold(text).chars().filter(|c| !c.is_whitespace()).collect();
    // The title is long and small: a letter is often lost ("FIAL"), so the
    // longer words may be off by one letter; short ones must match exactly.
    let kind = [
        ("BUTUNLEME", "Bütünleme"),
        ("MAZERET", "Mazeret"),
        ("FINAL", "Final"),
        ("ARASINAV", "Vize"),
        ("VIZE", "Vize"),
    ]
    .iter()
    .find(|(key, _)| contains_roughly(&folded, key, usize::from(key.len() >= 5)))
    .map(|(_, kind)| (*kind).to_owned());
    let term = [("BAHAR", "Bahar"), ("GUZ", "Güz"), ("YAZ", "Yaz")]
        .iter()
        .find(|(key, _)| folded.contains(key))
        .map(|(_, term)| (*term).to_owned());
    let year = folded.char_indices().find_map(|(i, _)| {
        let window = folded.get(i..i + 9)?;
        let (a, b) = (window.get(0..4)?, window.get(5..9)?);
        (window.as_bytes()[4] == b'-'
            && a.starts_with("20")
            && a.chars().all(|c| c.is_ascii_digit())
            && b.chars().all(|c| c.is_ascii_digit()))
        .then(|| format!("{a}-{b}"))
    });
    Title { kind, year, term }
}

/// Whether `text` contains `word` with at most `errors` letters added,
/// dropped or changed.
fn contains_roughly(text: &str, word: &str, errors: usize) -> bool {
    let text: Vec<char> = text.chars().collect();
    // Edit distance of `word` against the best matching part of `text`.
    let mut previous = vec![0; text.len() + 1];
    for (i, w) in word.chars().enumerate() {
        let mut current = vec![i + 1; text.len() + 1];
        for (j, t) in text.iter().enumerate() {
            current[j + 1] = (previous[j] + usize::from(w != *t))
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        previous = current;
    }
    previous.into_iter().min().unwrap_or(usize::MAX) <= errors
}

/// `-SINAV SORULARI -- GRUP A-` → `A`.
pub fn parse_group(text: &str) -> Option<String> {
    let folded: String = fold(text).chars().filter(|c| !c.is_whitespace()).collect();
    let after = folded.split("GRUP").nth(1)?;
    after
        .chars()
        .find(|c| c.is_ascii_uppercase())
        .map(|c| c.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_labels_are_read_with_the_usual_confusions() {
        assert_eq!(parse_mapping("(S1)- PÇ2"), Some((1, vec![2])));
        assert_eq!(parse_mapping(" (SI)-PC2"), Some((1, vec![2])));
        assert_eq!(parse_mapping("(S2)-PC1"), Some((2, vec![1])));
        assert_eq!(parse_mapping("(S10)-PÇ11"), Some((10, vec![11])));
        assert_eq!(parse_mapping("(S3)-PÇ1,PÇ4"), Some((3, vec![1, 4])));
        assert_eq!(parse_mapping("Puan"), None);
    }

    #[test]
    fn course_code_and_name_are_split() {
        assert_eq!(
            parse_course("BiLM374-2022-YAPAY ZEKA"),
            (Some("BİLM374".to_owned()), Some("YAPAY ZEKA".to_owned()))
        );
        assert_eq!(
            parse_course("MAT101-ANALİZ"),
            (Some("MAT101".to_owned()), Some("ANALİZ".to_owned()))
        );
        assert_eq!(parse_course(""), (None, None));
    }

    #[test]
    fn exam_kind_year_and_term_come_from_the_title() {
        let title = parse_title("2025-06 EGITI0GRETIYL AARDONEMi FINAL SNAVKAG");
        assert_eq!(title.kind.as_deref(), Some("Final"));
        let title = parse_title("2025 -2026 EGITM-OGRETIM YLBAHARDONEMi FINAL SNA KAGDI");
        assert_eq!(title.kind.as_deref(), Some("Final"));
        assert_eq!(title.year.as_deref(), Some("2025-2026"));
        assert_eq!(title.term.as_deref(), Some("Bahar"));
        assert_eq!(
            parse_title("2024-2025 GÜZ DÖNEMİ VİZE SINAV KÂĞIDI")
                .kind
                .as_deref(),
            Some("Vize")
        );
    }

    #[test]
    fn a_lost_letter_in_the_title_is_tolerated_for_long_words_only() {
        let title = parse_title("2025-2026 EGTM-ORETIMYL BAHARDONEMiFIAL SINAKAGD");
        assert_eq!(title.kind.as_deref(), Some("Final"));
        assert_eq!(title.year.as_deref(), Some("2025-2026"));
        assert_eq!(
            parse_title("BUTUNLEMESINAV").kind.as_deref(),
            Some("Bütünleme")
        );
        assert_eq!(parse_title("VIZF SINAV").kind, None);
    }

    #[test]
    fn group_letter_is_read() {
        assert_eq!(
            parse_group("-SINAV SORULARI -- GRUP A-").as_deref(),
            Some("A")
        );
        assert_eq!(parse_group("SINAV SORULARI GRUPB").as_deref(), Some("B"));
        assert_eq!(parse_group("SINAV SORULARI"), None);
    }

    #[test]
    fn line_runs_are_merged_into_peaks() {
        let values = [0.0, 0.9, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1];
        assert_eq!(peaks(&values, 0.7, 2), vec![1, 8]);
    }
}
