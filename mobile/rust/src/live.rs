//! Live scanning: sheets are read straight from the camera preview, one after
//! the other, without taking photos.
//!
//! Each frame the sheet is tracked from where it was in the previous frame (a
//! full search only when it was lost), its cells are read, and the readings of
//! the last few frames are combined. As soon as the combined reading is certain
//! (every score and the total accepted, the total adds up, the student number
//! clear) the sheet is saved. It stays saved while it is in view, and when it
//! is shown again; the next sheet is recognised by its different handwriting
//! (what is read can differ from frame to frame, the ink in the cells hardly).

use crate::api::camera::{CameraFrame, FrameFormat, ScanState, ScanUpdate};
use crate::api::reader::{ExamReader, SheetResult, NUMBER_FIELD};
use anyhow::{anyhow, bail};
use exam_infrastructure::sheet::{fuse_readings, SheetReading};
use exam_infrastructure::vision::warp_quad;
use exam_infrastructure::vision::Point;
use image::RgbImage;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Readings combined at most (the newest frames).
const WINDOW: usize = 4;
/// Frames without a sheet before tracking stops and the readings are dropped.
const LOST_AFTER: u32 = 3;
/// Frames a sheet may stay uncertain before it is saved for review anyway
/// (about a second), so the instructor is never kept waiting on a sheet whose
/// values cannot all be read with certainty, e.g. a total that does not add up.
const PATIENCE: u32 = 5;
/// Size each handwritten cell is reduced to for [`ink_signature`].
const SIGNATURE_CELL: (u32, u32) = (16, 6);
/// Likeness of two signatures from which they are the same sheet (another
/// sheet's handwriting gives about 0.1, the same sheet moved and blurred 0.6
/// and more).
const SAME_INK: f32 = 0.4;
/// Combined confidence the student number needs before a sheet is saved.
const NUMBER_CONFIDENCE: f32 = 0.8;
/// Review reasons that mean the values cannot be right yet.
const BLOCKING: [&str; 3] = [
    "total_mismatch",
    "score_exceeds_maximum",
    "ambiguous_corrected_score",
];

#[derive(Default)]
pub(crate) struct LiveState {
    previous: Option<[Point; 4]>,
    misses: u32,
    recent: Vec<SheetReading>,
    /// Handwriting of the sheet whose frames are in `recent`.
    ink: Option<Vec<f32>>,
    /// Frames the sheet in view has been read without being saved.
    reading_for: u32,
    /// Every sheet saved while scanning, so that one shown again is not
    /// saved twice.
    saved: Vec<Saved>,
}

/// What identifies the last saved sheet when it is seen again.
struct Saved {
    ink: Vec<f32>,
}

pub(crate) fn scan(reader: &ExamReader, frame: CameraFrame) -> anyhow::Result<ScanUpdate> {
    let started = Instant::now();
    let rgb = frame_to_rgb(&frame)?;
    let mut state = reader
        .live
        .lock()
        .map_err(|_| anyhow!("live scan state is poisoned"))?;
    let update = step(reader, &mut state, &rgb)?;
    Ok(ScanUpdate {
        millis: started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
        ..update
    })
}

fn update(
    state: ScanState,
    table: Vec<f32>,
    sheet: Option<SheetResult>,
    frames: usize,
) -> ScanUpdate {
    ScanUpdate {
        state,
        table,
        sheet,
        verified: false,
        frames: frames as u32,
        millis: 0,
    }
}

fn step(reader: &ExamReader, state: &mut LiveState, rgb: &RgbImage) -> anyhow::Result<ScanUpdate> {
    let located = reader.analyzer.locate(rgb, state.previous.as_ref());
    let mut seen = crate::diagnostics::Seen {
        image: rgb.clone(),
        source: "kamera",
        page_quad: Some(located.page_quad),
        registration_score: located.registration_score,
        table: reader.analyzer.table_outline(&located),
        reading: None,
    };
    if !located.found() {
        reader.remember(seen);
        state.misses += 1;
        if state.misses >= LOST_AFTER {
            state.previous = None;
            state.recent.clear();
            state.ink = None;
            state.reading_for = 0;
        }
        return Ok(update(ScanState::Searching, Vec::new(), None, 0));
    }
    state.previous = Some(located.page_quad);
    state.misses = 0;
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let table: Vec<f32> = reader
        .analyzer
        .table_outline(&located)
        .map(|quad| {
            quad.iter()
                .flat_map(|p| [p.x as f32 / w, p.y as f32 / h])
                .collect()
        })
        .unwrap_or_default();

    let reading = reader
        .analyzer
        .read_at(rgb, &located)
        .map_err(|e| anyhow!("{e}"))?;
    seen.reading = Some(reading.clone());
    reader.remember(seen);
    // Different handwriting means another sheet: its frames must not be
    // mixed with the previous sheet's. (Not a different student number: one
    // blurred frame reads it differently, and the frames gathered so far
    // would be thrown away.)
    let ink = ink_signature(rgb, &reading);
    let other_sheet = state
        .ink
        .as_ref()
        .is_some_and(|before| likeness(before, &ink) < SAME_INK);
    if other_sheet {
        state.recent.clear();
        state.reading_for = 0;
    }
    if other_sheet || state.ink.is_none() {
        state.ink = Some(ink.clone());
    }
    state.recent.push(reading);
    if state.recent.len() > WINDOW {
        state.recent.remove(0);
    }
    let fused = fuse_readings(&state.recent).expect("at least one reading");
    let frames = state.recent.len();

    if state
        .saved
        .iter()
        .any(|saved| likeness(&saved.ink, &ink) >= SAME_INK)
    {
        return Ok(update(ScanState::AlreadySaved, table, None, frames));
    }
    let id = sheet_id();
    let result = reader.sheet_result(id.clone(), rgb, &fused, false)?;

    state.reading_for += 1;
    let verified = certain(&result);
    if verified || state.reading_for >= PATIENCE {
        let result = reader.sheet_result(id, rgb, &fused, true)?;
        state.saved.push(Saved {
            ink: state.ink.clone().unwrap_or(ink),
        });
        state.recent.clear();
        state.reading_for = 0;
        return Ok(ScanUpdate {
            verified,
            ..update(ScanState::Saved, table, Some(result), frames)
        });
    }
    Ok(update(ScanState::Reading, table, None, frames))
}

/// Every score and the total accepted and consistent, the number clear.
fn certain(result: &SheetResult) -> bool {
    let number_clear = result
        .cells
        .iter()
        .find(|c| c.field == NUMBER_FIELD)
        .is_some_and(|c| c.value.is_some() && c.confidence >= NUMBER_CONFIDENCE);
    number_clear
        && result
            .cells
            .iter()
            .filter(|c| c.field != NUMBER_FIELD)
            .all(|c| c.accepted)
        && !result
            .issues
            .iter()
            .any(|i| BLOCKING.contains(&i.reason.as_str()))
}

/// Numbers that differ in at most one digit (a misread) count as the same.
#[cfg(test)]
fn similar_numbers(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return true;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            current.push(
                (current[j] + 1)
                    .min(previous[j + 1] + 1)
                    .min(previous[j] + usize::from(ca != cb)),
            );
        }
        previous = current;
    }
    previous[b.len()] <= 1
}

/// The ink in the sheet's handwritten cells (number, scores, total), each
/// reduced to a few pixels: the same sheet gives about the same signature in
/// every frame, however its digits are read; another sheet does not.
fn ink_signature(rgb: &RgbImage, reading: &SheetReading) -> Vec<f32> {
    let (w, h) = SIGNATURE_CELL;
    let mut signature = Vec::new();
    for field in reading.fields.iter().filter(|f| f.reading.is_some()) {
        let cell = warp_quad(rgb, &field.quad, w, h);
        // The weakest channel: red, blue and black ink are all dark in it.
        let values: Vec<f32> = cell
            .pixels()
            .map(|p| f32::from(p[0].min(p[1]).min(p[2])))
            .collect();
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        let paper = sorted[sorted.len() * 9 / 10].max(1.0);
        signature.extend(values.iter().map(|v| (1.0 - v / paper).clamp(0.0, 1.0)));
    }
    signature
}

/// Correlation of two ink signatures; blank cells on both count as alike.
fn likeness(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x - ma, y - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    // Almost no ink: nothing written yet on either, or on one only.
    const BLANK: f32 = 1e-3;
    if aa / n < BLANK || bb / n < BLANK {
        return if aa / n < BLANK && bb / n < BLANK {
            1.0
        } else {
            0.0
        };
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn sheet_id() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros().to_string())
        .unwrap_or_else(|_| "0".to_owned())
}

/// The frame as an upright RGB image.
pub(crate) fn frame_to_rgb(frame: &CameraFrame) -> anyhow::Result<RgbImage> {
    let (width, height) = (frame.width as usize, frame.height as usize);
    if width == 0 || height == 0 {
        bail!("empty camera frame");
    }
    let plane = |index: usize| {
        frame
            .planes
            .get(index)
            .ok_or_else(|| anyhow!("camera frame has no plane {index}"))
    };
    let mut rgb = vec![0_u8; width * height * 3];
    match frame.format {
        FrameFormat::Yuv420 => {
            let (y, u, v) = (plane(0)?, plane(1)?, plane(2)?);
            for row in 0..height {
                for col in 0..width {
                    let luma = f32::from(
                        *y.bytes
                            .get(row * y.row_stride as usize + col * y.pixel_stride.max(1) as usize)
                            .ok_or_else(|| anyhow!("Y plane too small"))?,
                    );
                    let chroma = |p: &crate::api::camera::FramePlane| {
                        p.bytes
                            .get(
                                (row / 2) * p.row_stride as usize
                                    + (col / 2) * p.pixel_stride.max(1) as usize,
                            )
                            .map(|&b| f32::from(b) - 128.0)
                            .ok_or_else(|| anyhow!("chroma plane too small"))
                    };
                    let (cb, cr) = (chroma(u)?, chroma(v)?);
                    let out = (row * width + col) * 3;
                    rgb[out] = (luma + 1.402 * cr).clamp(0.0, 255.0) as u8;
                    rgb[out + 1] = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 255.0) as u8;
                    rgb[out + 2] = (luma + 1.772 * cb).clamp(0.0, 255.0) as u8;
                }
            }
        }
        FrameFormat::Bgra8888 | FrameFormat::Rgba8888 => {
            let p = plane(0)?;
            let order = if matches!(frame.format, FrameFormat::Bgra8888) {
                [2, 1, 0]
            } else {
                [0, 1, 2]
            };
            for row in 0..height {
                let start = row * p.row_stride as usize;
                let line = p
                    .bytes
                    .get(start..start + width * 4)
                    .ok_or_else(|| anyhow!("camera frame too small"))?;
                for col in 0..width {
                    let out = (row * width + col) * 3;
                    for c in 0..3 {
                        rgb[out + c] = line[col * 4 + order[c]];
                    }
                }
            }
        }
    }
    let image = RgbImage::from_raw(frame.width, frame.height, rgb).expect("buffer size matches");
    Ok(match frame.rotation % 360 {
        90 => image::imageops::rotate90(&image),
        180 => image::imageops::rotate180(&image),
        270 => image::imageops::rotate270(&image),
        _ => image,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::camera::FramePlane;
    use crate::api::reader::MaximumScore;

    fn repo(path: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    }

    fn reader() -> ExamReader {
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        ExamReader::new(
            String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
            read("tools/synthetic/templates/inonu-final-a.png"),
            read("models/digit-crnn-v4.onnx"),
            read("tools/synthetic/templates/inonu-final-a.docx"),
            ["S3", "S4"]
                .map(|q| MaximumScore {
                    question: q.into(),
                    value: "25".into(),
                })
                .into(),
            Vec::new(),
        )
        .unwrap()
    }

    fn sheet(index: usize) -> RgbImage {
        let path = format!(
            "crates/exam_infrastructure/tests/fixtures/sheets/images/syn-900-{index:05}.jpg"
        );
        image::open(repo(&path)).unwrap().to_rgb8()
    }

    /// An RGBA frame of `image` moved by `(dx, dy)`, like a hand-held phone.
    fn frame(image: &RgbImage, dx: i64, dy: i64) -> CameraFrame {
        let mut moved =
            RgbImage::from_pixel(image.width(), image.height(), image::Rgb([90, 90, 90]));
        image::imageops::overlay(&mut moved, image, dx, dy);
        let rgba: Vec<u8> = moved
            .pixels()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        CameraFrame {
            width: moved.width(),
            height: moved.height(),
            format: FrameFormat::Rgba8888,
            planes: vec![FramePlane {
                bytes: rgba,
                row_stride: moved.width() * 4,
                pixel_stride: 4,
            }],
            rotation: 0,
        }
    }

    fn empty_frame() -> CameraFrame {
        frame(
            &RgbImage::from_pixel(900, 1200, image::Rgb([90, 90, 90])),
            0,
            0,
        )
    }

    const JITTER: [(i64, i64); 6] = [(0, 0), (3, -2), (-2, 4), (5, 1), (1, -3), (-4, 2)];

    #[test]
    fn sheets_shown_one_after_another_are_each_saved_once() {
        let reader = reader();
        let mut states = Vec::new();
        let mut saved = Vec::new();
        let mut run = |frame: CameraFrame| {
            let update = reader.scan_frame(frame).unwrap();
            states.push(update.state);
            if let Some(sheet) = update.sheet {
                saved.push(sheet);
            }
            update.table.len()
        };
        run(empty_frame());
        let (a, b) = (sheet(2), sheet(0));
        for (dx, dy) in JITTER {
            assert_eq!(run(frame(&a, dx, dy)), 8, "table outline");
        }
        for _ in 0..LOST_AFTER {
            run(empty_frame());
        }
        for (dx, dy) in JITTER {
            run(frame(&b, dx, dy));
        }

        let count = |s: ScanState| states.iter().filter(|&&x| x == s).count();
        assert_eq!(count(ScanState::Saved), 2, "{states:?}");
        assert_eq!(states[0], ScanState::Searching);
        assert!(states.contains(&ScanState::AlreadySaved), "{states:?}");
        // The right sheets, in order (a misread digit is the recognizer's and
        // is caught by the roster match, not by live scanning).
        let numbers: Vec<String> = saved
            .iter()
            .map(|s| {
                s.cells
                    .iter()
                    .find(|c| c.field == NUMBER_FIELD)
                    .and_then(|c| c.value.clone())
                    .unwrap_or_default()
            })
            .collect();
        assert!(similar_numbers(&numbers[0], "09111750633"), "{numbers:?}");
        assert!(similar_numbers(&numbers[1], "08028915324"), "{numbers:?}");
        assert!(saved
            .iter()
            .all(|s| s.cells.iter().all(|c| !c.crop_png.is_empty())));
    }

    #[test]
    fn tracking_is_much_faster_than_finding_the_sheet() {
        let reader = reader();
        let a = sheet(1);
        let first = reader.scan_frame(frame(&a, 0, 0)).unwrap();
        let tracked: Vec<u32> = JITTER[1..4]
            .iter()
            .map(|&(dx, dy)| reader.scan_frame(frame(&a, dx, dy)).unwrap().millis)
            .collect();
        eprintln!(
            "first frame {} ms, tracked frames {tracked:?} ms",
            first.millis
        );
        assert!(
            tracked.iter().all(|&t| t < first.millis),
            "{} vs {tracked:?}",
            first.millis
        );
    }

    fn yuv_frame(image: &RgbImage, interleaved: bool) -> CameraFrame {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut y = vec![0_u8; w * h];
        let mut u = vec![0_u8; cw * ch];
        let mut v = vec![0_u8; cw * ch];
        for (x, row, p) in image.enumerate_pixels() {
            let (r, g, b) = (f32::from(p[0]), f32::from(p[1]), f32::from(p[2]));
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            y[row as usize * w + x as usize] = luma.round() as u8;
            if x % 2 == 0 && row % 2 == 0 {
                let i = (row as usize / 2) * cw + x as usize / 2;
                u[i] = ((b - luma) / 1.772 + 128.0).round().clamp(0.0, 255.0) as u8;
                v[i] = ((r - luma) / 1.402 + 128.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let plane = |bytes: Vec<u8>, row_stride: usize, pixel_stride: u32| FramePlane {
            bytes,
            row_stride: row_stride as u32,
            pixel_stride,
        };
        let planes = if interleaved {
            // Android NV21-style: V and U share one buffer, pixel stride 2.
            let mut vu = vec![0_u8; cw * ch * 2];
            for i in 0..cw * ch {
                vu[2 * i] = v[i];
                vu[2 * i + 1] = u[i];
            }
            let u_bytes = vu[1..].to_vec();
            vec![
                plane(y, w, 1),
                plane(u_bytes, cw * 2, 2),
                plane(vu, cw * 2, 2),
            ]
        } else {
            vec![plane(y, w, 1), plane(u, cw, 1), plane(v, cw, 1)]
        };
        CameraFrame {
            width: w as u32,
            height: h as u32,
            format: FrameFormat::Yuv420,
            planes,
            rotation: 0,
        }
    }

    #[test]
    fn android_yuv_frames_become_the_same_picture() {
        let colours = [
            image::Rgb([200, 30, 40]),
            image::Rgb([20, 180, 60]),
            image::Rgb([240, 240, 240]),
        ];
        // 2x2 blocks of one colour, since 2x2 pixels share their chroma.
        let image = RgbImage::from_fn(8, 6, |x, y| colours[((x / 2 + y / 2) % 3) as usize]);
        for interleaved in [false, true] {
            let rgb = frame_to_rgb(&yuv_frame(&image, interleaved)).unwrap();
            for (a, b) in image.pixels().zip(rgb.pixels()) {
                for c in 0..3 {
                    assert!(
                        (i16::from(a[c]) - i16::from(b[c])).abs() <= 3,
                        "{a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ios_bgra_frames_are_reordered_and_rotated_upright() {
        // A 2x1 sensor image (red, blue) turned 90° clockwise is 1x2, red on top.
        let frame = CameraFrame {
            width: 2,
            height: 1,
            format: FrameFormat::Bgra8888,
            planes: vec![FramePlane {
                bytes: vec![0, 0, 255, 255, 255, 0, 0, 255, 9, 9, 9, 9],
                row_stride: 12,
                pixel_stride: 4,
            }],
            rotation: 90,
        };
        let rgb = frame_to_rgb(&frame).unwrap();
        assert_eq!(rgb.dimensions(), (1, 2));
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(0, 1).0, [0, 0, 255]);
    }

    #[test]
    fn the_last_frame_can_be_shared_as_a_diagnostic() {
        let reader = reader();
        assert!(reader.last_diagnostic().unwrap().is_none());
        reader.scan_frame(empty_frame()).unwrap();
        let empty = reader.last_diagnostic().unwrap().unwrap();
        assert!(
            empty.report.contains("Kâğıt bulundu: hayır"),
            "{}",
            empty.report
        );

        reader.scan_frame(frame(&sheet(1), 0, 0)).unwrap();
        let found = reader.last_diagnostic().unwrap().unwrap();
        assert!(
            found.report.contains("Kâğıt bulundu: evet"),
            "{}",
            found.report
        );
        assert!(found.report.contains("question:1.earned_score"));
        let picture = image::load_from_memory(&found.image_png).unwrap();
        assert!(picture.width().max(picture.height()) <= 1600);
        if let Ok(path) = std::env::var("DIAGNOSTIC_PNG") {
            std::fs::write(path, &found.image_png).unwrap();
        }
    }

    #[test]
    fn numbers_one_digit_apart_are_the_same_student() {
        assert!(similar_numbers("04451095728", "04451095728"));
        assert!(similar_numbers("04451095728", "04451095723"));
        assert!(similar_numbers("04451095728", "0445109572"));
        assert!(!similar_numbers("04451095728", "09111750633"));
        assert!(similar_numbers("", "09111750633"));
    }

    /// Signature of the cells of `image` moved by `(dx, dy)` and blurred.
    fn signature(reader: &ExamReader, image: &RgbImage, dx: i64, dy: i64, blur: f32) -> Vec<f32> {
        let rgb = frame_to_rgb(&frame(image, dx, dy)).unwrap();
        let rgb = if blur > 0.0 {
            image::imageops::blur(&rgb, blur)
        } else {
            rgb
        };
        let located = reader.analyzer.locate(&rgb, None);
        let reading = reader.analyzer.read_at(&rgb, &located).unwrap();
        ink_signature(&rgb, &reading)
    }

    #[test]
    fn the_same_sheet_has_the_same_handwriting_and_another_does_not() {
        let reader = reader();
        let sheets = [sheet(0), sheet(1), sheet(2)];
        for (index, image) in sheets.iter().enumerate() {
            let still = signature(&reader, image, 0, 0, 0.0);
            for (dx, dy, blur) in [(6, -4, 0.0), (-3, 5, 1.5), (2, 2, 3.0)] {
                let moved = signature(&reader, image, dx, dy, blur);
                let alike = likeness(&still, &moved);
                assert!(
                    alike >= SAME_INK + 0.1,
                    "sheet {index} ({dx},{dy},{blur}): {alike}"
                );
            }
            for (other, image) in sheets.iter().enumerate().skip(index + 1) {
                let alike = likeness(&still, &signature(&reader, image, 0, 0, 0.0));
                assert!(alike < SAME_INK - 0.2, "sheets {index}/{other}: {alike}");
            }
        }
    }

    #[test]
    fn a_sheet_swapped_in_place_is_saved_and_one_shown_again_is_not() {
        let reader = reader();
        let mut saved = 0;
        let mut states = Vec::new();
        let mut run = |frame: CameraFrame| {
            let update = reader.scan_frame(frame).unwrap();
            states.push(update.state);
            saved += usize::from(update.sheet.is_some());
        };
        let (a, b) = (sheet(2), sheet(0));
        // The next sheet laid on the previous one: never out of view.
        for (dx, dy) in JITTER {
            run(frame(&a, dx, dy));
        }
        for (dx, dy) in JITTER {
            run(frame(&b, dx, dy));
        }
        // The first sheet shown again after the camera looked away.
        for _ in 0..LOST_AFTER {
            run(empty_frame());
        }
        for (dx, dy) in JITTER {
            run(frame(&a, dx, dy));
        }
        assert_eq!(saved, 2, "{states:?}");
    }
}
