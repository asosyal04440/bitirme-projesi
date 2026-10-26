//! Recognising which exam a sheet belongs to from the camera, before any
//! exam is set up: the instructor just shows the first sheet.

use crate::api::camera::CameraFrame;
use crate::api::definition::ExamDefinition;
use anyhow::anyhow;
use exam_import::{ExamMetadata, ImportedExam};
use exam_infrastructure::detect::detect_exam;
use exam_infrastructure::printed::PrintedTextReader;
use exam_infrastructure::sheet::SheetTemplate;
use exam_infrastructure::vision::Point;
use std::sync::Mutex;
use std::time::Instant;

/// Readings of the exam kept at most (the newest frames).
const WINDOW: usize = 5;
/// Frames without a sheet before the readings are dropped.
const LOST_AFTER: u32 = 3;

#[flutter_rust_bridge::frb(opaque)]
pub struct SheetScanner {
    template: SheetTemplate,
    text: PrintedTextReader,
    state: Mutex<DetectState>,
}

#[flutter_rust_bridge::frb(ignore)]
#[derive(Default)]
struct DetectState {
    previous: Option<[Point; 4]>,
    misses: u32,
    readings: Vec<ImportedExam>,
}

pub struct DetectUpdate {
    /// True while a sheet of the form is in view.
    pub found: bool,
    /// Corners of the Soru-PÇ table as `[x0, y0, … x3, y3]`, normalised to
    /// the upright frame; empty when no sheet is in view.
    pub table: Vec<f32>,
    /// The exam read off the sheet, once two frames agree on it (or the
    /// readings of several frames combined when they never fully agree).
    pub definition: Option<ExamDefinition>,
    /// Frames read so far for the sheet in view.
    pub attempts: u32,
    pub millis: u32,
}

impl SheetScanner {
    /// `ocr_model` and `ocr_keys` are PaddleOCR's recognition model and its
    /// character list, shipped with the app.
    pub fn new(
        template_json: String,
        template_png: Vec<u8>,
        ocr_model: Vec<u8>,
        ocr_keys: String,
    ) -> anyhow::Result<SheetScanner> {
        Ok(SheetScanner {
            template: SheetTemplate::from_bytes(&template_json, &template_png)
                .map_err(|e| anyhow!("{e}"))?,
            text: PrintedTextReader::from_bytes(ocr_model, &ocr_keys)
                .map_err(|e| anyhow!("{e}"))?,
            state: Mutex::new(DetectState::default()),
        })
    }

    /// Looks for a sheet in one camera frame and reads which exam it is.
    pub fn detect_frame(&self, frame: CameraFrame) -> anyhow::Result<DetectUpdate> {
        let started = Instant::now();
        let rgb = crate::live::frame_to_rgb(&frame)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("scanner state is poisoned"))?;
        let millis =
            |started: Instant| started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
        let located = self.template.locate(&rgb, state.previous.as_ref());
        if !located.found() {
            state.misses += 1;
            if state.misses >= LOST_AFTER {
                *state = DetectState::default();
            }
            return Ok(DetectUpdate {
                found: false,
                table: Vec::new(),
                definition: None,
                attempts: 0,
                millis: millis(started),
            });
        }
        if !located.tracked {
            state.readings.clear();
        }
        state.previous = Some(located.page_quad);
        state.misses = 0;
        let (w, h) = (rgb.width() as f32, rgb.height() as f32);
        let table = self
            .template
            .to_photo(&located)
            .map(|to_photo| {
                self.template
                    .table_rect()
                    .corners()
                    .iter()
                    .flat_map(|p| {
                        let q = to_photo.apply(*p);
                        [q.x as f32 / w, q.y as f32 / h]
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Ok(exam) = detect_exam(&self.template, &self.text, &rgb, &located) {
            state.readings.push(exam);
            if state.readings.len() > WINDOW {
                state.readings.remove(0);
            }
        }
        let definition = decide(&state.readings).map(|exam| ExamDefinition::from_imported(&exam));
        Ok(DetectUpdate {
            found: true,
            table,
            definition,
            attempts: state.readings.len() as u32,
            millis: millis(started),
        })
    }

    /// Prepares the text reader (builds its model plans), which takes a
    /// while on a phone; call it in the background when the app starts so the
    /// first sheet is recognised without that wait.
    pub fn warm_up(&self) -> anyhow::Result<()> {
        self.text.warm_up().map_err(|e| anyhow!("{e}"))
    }

    /// Forgets the sheet being read.
    #[flutter_rust_bridge::frb(sync)]
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = DetectState::default();
        }
    }
}

/// The most frequent value and how often it was seen.
fn vote<T: PartialEq + Clone>(values: impl Iterator<Item = T>) -> Option<(T, usize)> {
    let mut counts: Vec<(T, usize)> = Vec::new();
    for value in values {
        match counts.iter_mut().find(|(v, _)| *v == value) {
            Some((_, n)) => *n += 1,
            None => counts.push((value, 1)),
        }
    }
    counts.into_iter().max_by_key(|(_, n)| *n)
}

/// Combines the readings of consecutive frames field by field: the number
/// of questions, each question's label and PÇ, the course code and the exam
/// kind each count once two frames agree on them, so one frame that misses a
/// field does not hold the others back. After a full window the most
/// frequent value of each field is taken (gaps are left for the
/// instructor to fill in).
fn decide(readings: &[ImportedExam]) -> Option<ImportedExam> {
    let final_call = readings.len() >= WINDOW;
    let (count, seen) = vote(readings.iter().map(|e| e.questions.len()))?;
    if seen < 2 && !final_call {
        return None;
    }
    let same: Vec<&ImportedExam> = readings
        .iter()
        .filter(|e| e.questions.len() == count)
        .collect();
    let mut exam = (*same.last()?).clone();
    let mut agreed = true;
    for (index, question) in exam.questions.iter_mut().enumerate() {
        let column = || same.iter().map(|e| &e.questions[index]);
        if let Some((id, n)) = vote(column().map(|q| (q.printed_id.clone(), q.id.clone()))) {
            (question.printed_id, question.id) = id;
            agreed &= n >= 2;
        }
        match vote(
            column()
                .map(|q| q.outcomes.clone())
                .filter(|o| !o.is_empty()),
        ) {
            Some((outcomes, n)) => {
                question.outcomes = outcomes;
                agreed &= n >= 2;
            }
            None => agreed = false,
        }
    }
    let field = |get: fn(&ExamMetadata) -> &Option<String>| {
        vote(same.iter().filter_map(|e| get(&e.metadata).clone()))
    };
    let m = &mut exam.metadata;
    for (target, get, required) in [
        (
            &mut m.course_code,
            (|m| &m.course_code) as fn(&ExamMetadata) -> &Option<String>,
            true,
        ),
        (&mut m.exam_kind, |m| &m.exam_kind, true),
        (&mut m.course_name, |m| &m.course_name, false),
        (&mut m.course, |m| &m.course, false),
        (&mut m.academic_year, |m| &m.academic_year, false),
        (&mut m.term, |m| &m.term, false),
        (&mut m.group, |m| &m.group, false),
    ] {
        let voted = field(get);
        if required {
            agreed &= voted.as_ref().is_some_and(|(_, n)| *n >= 2);
        }
        *target = voted.map(|(v, _)| v);
    }
    (agreed || final_call).then_some(exam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::camera::{FrameFormat, FramePlane, ScanState};
    use crate::api::reader::{ExamReader, MaximumScore};

    fn repo(path: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    }

    fn read(path: &str) -> Vec<u8> {
        std::fs::read(repo(path)).unwrap()
    }

    fn frame(path: &str) -> CameraFrame {
        let image = image::open(repo(path)).unwrap().to_rgb8();
        CameraFrame {
            width: image.width(),
            height: image.height(),
            format: FrameFormat::Rgba8888,
            planes: vec![FramePlane {
                bytes: image
                    .pixels()
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect(),
                row_stride: image.width() * 4,
                pixel_stride: 4,
            }],
            rotation: 0,
        }
    }

    const SHEET: &str = "crates/exam_infrastructure/tests/fixtures/detect/clean-a.jpg";

    #[test]
    fn the_first_sheet_sets_up_the_exam_and_is_then_read_without_a_word_file() {
        let template_json =
            String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap();
        let template_png = read("tools/synthetic/templates/inonu-final-a.png");
        let scanner = SheetScanner::new(
            template_json.clone(),
            template_png.clone(),
            read("models/ppocr-v4-rec.onnx"),
            String::from_utf8(read("models/ppocr-v4-rec.keys.txt")).unwrap(),
        )
        .unwrap();
        let mut definition = None;
        for _ in 0..WINDOW {
            let update = scanner.detect_frame(frame(SHEET)).unwrap();
            eprintln!(
                "detect frame: {} ms, {} readings",
                update.millis, update.attempts
            );
            assert!(update.found);
            if update.definition.is_some() {
                definition = update.definition;
                break;
            }
        }
        let definition = definition.expect("exam recognised");
        assert_eq!(definition.course_code.as_deref(), Some("BİLM374"));
        assert_eq!(definition.exam_kind.as_deref(), Some("Final"));
        let outcomes: Vec<String> = definition
            .questions
            .iter()
            .map(|q| format!("{}-{}", q.printed_id, q.outcomes.join(",")))
            .collect();
        assert_eq!(outcomes, ["S1-PÇ2", "S2-PÇ1", "S3-PÇ3", "S4-PÇ2"]);

        // The instructor enters the maximum scores; sheets are then read.
        let reader = ExamReader::from_definition(
            template_json,
            template_png,
            read("models/digit-crnn-v4.onnx"),
            definition,
            [("S1", "20"), ("S2", "30"), ("S3", "25"), ("S4", "25")]
                .map(|(q, v)| MaximumScore {
                    question: q.into(),
                    value: v.into(),
                })
                .into(),
            Vec::new(),
        )
        .unwrap();
        let mut saved = None;
        for _ in 0..8 {
            let update = reader.scan_frame(frame(SHEET)).unwrap();
            if update.state == ScanState::Saved {
                saved = update.sheet;
                break;
            }
        }
        let sheet = saved.expect("sheet saved");
        let values: Vec<(String, Option<String>)> = sheet
            .cells
            .iter()
            .map(|c| (c.field.clone(), c.value.clone()))
            .collect();
        let expected = [
            ("Numara", "03310352098"),
            ("S1", "4"),
            ("S2", "14.5"),
            ("S3", "20"),
            ("S4", "18"),
            ("Not", "56.5"),
        ]
        .map(|(f, v)| (f.to_owned(), Some(v.to_owned())));
        assert_eq!(values, expected);
        assert_eq!(sheet.outcomes.len(), 3);
    }
}
