//! Live scanning measured the way video document recognition is (MIDV-500,
//! Arlazarov et al. 2019): each synthetic sheet is shown to the scanner as a
//! short clip of hand-held frames — moving, some blurred — and what is saved
//! is compared with the truth. Run with
//! `cargo test --release live_eval -- --ignored --nocapture`. Test recordings
//! made in the app are replayed the same way (`live_replay`).

use crate::api::camera::{CameraFrame, FrameFormat, FramePlane, ScanState};
use crate::api::reader::{ExamReader, MaximumScore};
use image::RgbImage;
use std::path::{Path, PathBuf};

fn repo(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

/// A small deterministic generator, so that every run sees the same clips.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn frame(image: &RgbImage, dx: i64, dy: i64, blur: f32) -> CameraFrame {
    let mut moved = RgbImage::from_pixel(image.width(), image.height(), image::Rgb([90, 90, 90]));
    image::imageops::overlay(&mut moved, image, dx, dy);
    let moved = if blur > 0.0 {
        image::imageops::blur(&moved, blur)
    } else {
        moved
    };
    CameraFrame {
        width: moved.width(),
        height: moved.height(),
        format: FrameFormat::Rgba8888,
        planes: vec![FramePlane {
            bytes: moved
                .pixels()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            row_stride: moved.width() * 4,
            pixel_stride: 4,
        }],
        rotation: 0,
    }
}

#[test]
#[ignore = "slow measurement; run on demand"]
fn live_eval() {
    let read = |p: &str| std::fs::read(repo(p)).unwrap();
    let reader = ExamReader::new(
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
    .unwrap();
    let sheets: usize = std::env::var("LIVE_EVAL_SHEETS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let labels = std::fs::read_to_string(repo("data/synthetic/e2e/shard-1/labels.jsonl")).unwrap();
    let mut rng = Lcg(7);
    let (mut cells, mut right, mut sheets_right, mut saved_sheets, mut frames_used) =
        (0, 0, 0, 0, 0);
    let mut duplicates = 0;
    let blank = RgbImage::from_pixel(1200, 1600, image::Rgb([90, 90, 90]));
    for line in labels.lines().take(sheets) {
        let label: serde_json::Value = serde_json::from_str(line).unwrap();
        let image = image::open(repo(&format!(
            "data/synthetic/e2e/shard-1/{}",
            label["image"].as_str().unwrap()
        )))
        .unwrap()
        .to_rgb8();
        for _ in 0..3 {
            reader.scan_frame(frame(&blank, 0, 0, 0.0)).unwrap();
        }
        // About two seconds of a hand-held phone: drifting a few pixels a
        // frame, every third frame or so blurred by motion or focus.
        let mut saved = Vec::new();
        let mut states = Vec::new();
        let (mut x, mut y) = (0.0_f64, 0.0_f64);
        for index in 0..14 {
            x += (rng.next() - 0.5) * 12.0;
            y += (rng.next() - 0.5) * 12.0;
            let blur = if rng.next() < 0.35 {
                1.0 + 3.0 * rng.next() as f32
            } else {
                0.0
            };
            let update = reader
                .scan_frame(frame(&image, x as i64, y as i64, blur))
                .unwrap();
            states.push(update.state);
            if update.state == ScanState::Saved {
                if saved.is_empty() {
                    frames_used += index + 1;
                }
                saved.extend(update.sheet);
            }
        }
        duplicates += saved.len().saturating_sub(1);
        let Some(sheet) = saved.first() else {
            println!("NOT SAVED {}: {states:?}", label["sample_id"]);
            continue;
        };
        saved_sheets += 1;
        let truth = |key: &str| {
            label["cells"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["field_key"] == key)
                .and_then(|c| c["value"].as_str())
                .unwrap_or("")
                .to_owned()
        };
        let mut all = true;
        for (field, key) in [
            ("Numara", "student.number".to_owned()),
            ("Not", "declared_total".to_owned()),
        ]
        .into_iter()
        .chain((1..=4).map(|q| {
            (
                ["S1", "S2", "S3", "S4"][q - 1],
                format!("question:{q}.earned_score"),
            )
        })) {
            let got = sheet
                .cells
                .iter()
                .find(|c| c.field == field)
                .and_then(|c| c.value.clone())
                .unwrap_or_default();
            cells += 1;
            if got == truth(&key) {
                right += 1;
            } else {
                all = false;
            }
        }
        sheets_right += usize::from(all);
    }
    println!(
        "LIVE_EVAL sheets {sheets} saved {saved_sheets} duplicates {duplicates} \
         cells {:.3} sheets_all_right {:.3} mean_frames_to_save {:.2}",
        right as f64 / cells.max(1) as f64,
        sheets_right as f64 / saved_sheets.max(1) as f64,
        frames_used as f64 / saved_sheets.max(1) as f64
    );
}

/// The exam as the app stores it (`definition_json.dart`).
fn definition_from_json(json: &serde_json::Value) -> crate::api::definition::ExamDefinition {
    let text = |key: &str| json[key].as_str().map(str::to_owned);
    let number = |value: &serde_json::Value| value.as_u64().unwrap_or(0) as u32;
    crate::api::definition::ExamDefinition {
        course: text("course"),
        course_code: text("courseCode"),
        course_name: text("courseName"),
        academic_year: text("academicYear"),
        term: text("term"),
        exam_kind: text("examKind"),
        group: text("group"),
        page_width: number(&json["pageWidth"]),
        total_left: number(&json["totalLeft"]),
        total_width: number(&json["totalWidth"]),
        questions: json["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| crate::api::definition::DefinedQuestion {
                printed_id: q["printedId"].as_str().unwrap().to_owned(),
                outcomes: q["outcomes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|o| o.as_str().map(str::to_owned))
                    .collect(),
                maximum: q["maximum"].as_str().map(str::to_owned),
                left: number(&q["left"]),
                width: number(&q["width"]),
            })
            .collect(),
    }
}

/// What replaying a test recording gave, compared with the instructor's
/// checked values.
#[derive(Debug, Default, PartialEq)]
struct Replayed {
    /// Sheets the app saved during the recording, and those confirmed.
    app_saved: usize,
    confirmed: usize,
    /// Sheets saved in the replay, and those that match a confirmed sheet.
    saved: usize,
    matched: usize,
    cells: usize,
    right: usize,
}

/// Feeds a recording's frames, in order, to `reader` (with the change being
/// measured) and scores what it saves against the confirmed sheets: each
/// sheet saved in the replay is matched with the confirmed sheet the app
/// saved nearest in frames.
fn replay(dir: &Path, reader: &ExamReader) -> Replayed {
    let lines = std::fs::read_to_string(dir.join("frames.jsonl")).unwrap();
    let frames: Vec<serde_json::Value> = lines
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let labels: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("labels.json")).unwrap()).unwrap();
    let truth = |sheet: &str| {
        labels
            .iter()
            .find(|l| l["sheet"] == sheet && l["confirmed"] == true)
    };
    let app: Vec<(usize, String)> = frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| Some((i, f["sheet"].as_str()?.to_owned())))
        .collect();
    let mut report = Replayed {
        app_saved: app.len(),
        confirmed: app.iter().filter(|(_, s)| truth(s).is_some()).count(),
        ..Replayed::default()
    };
    reader.reset_scan();
    for (index, entry) in frames.iter().enumerate() {
        let image = image::open(dir.join(entry["frame"].as_str().unwrap()))
            .unwrap()
            .to_rgb8();
        let update = reader.scan_frame(frame(&image, 0, 0, 0.0)).unwrap();
        let Some(sheet) = update.sheet else {
            continue;
        };
        report.saved += 1;
        let Some((_, app_sheet)) = app
            .iter()
            .filter(|(at, _)| at.abs_diff(index) <= 15)
            .min_by_key(|(at, _)| at.abs_diff(index))
        else {
            continue;
        };
        let Some(label) = truth(app_sheet) else {
            continue;
        };
        report.matched += 1;
        for cell in &sheet.cells {
            report.cells += 1;
            let expected = label["values"][&cell.field].as_str().map(str::to_owned);
            report.right += usize::from(cell.value == expected);
        }
    }
    report
}

/// Replays a test recording shared from the app, unzipped into the folder
/// named by `RECORDING`:
/// `RECORDING=… cargo test --release live_replay -- --ignored --nocapture`.
#[test]
#[ignore = "needs a recording; run on demand"]
fn live_replay() {
    let dir = PathBuf::from(std::env::var("RECORDING").expect("RECORDING=<unzipped folder>"));
    let definition: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("definition.json")).unwrap())
            .unwrap();
    let read = |p: &str| std::fs::read(repo(p)).unwrap();
    let reader = ExamReader::from_definition(
        String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
        read("tools/synthetic/templates/inonu-final-a.png"),
        read("models/digit-crnn-v4.onnx"),
        definition_from_json(&definition),
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let report = replay(&dir, &reader);
    println!(
        "LIVE_REPLAY {report:?} cells {:.3}",
        report.right as f64 / report.cells.max(1) as f64
    );
}

#[test]
fn a_test_recording_replays_against_the_checked_values() {
    // Two synthetic sheets recorded the way the app does, what was saved
    // confirmed as it is, then replayed: the same reader saves and reads the
    // same, and the recording packs into a zip.
    let read = |p: &str| std::fs::read(repo(p)).unwrap();
    let reader = ExamReader::new(
        String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
        read("tools/synthetic/templates/inonu-final-a.png"),
        read("models/digit-crnn-v4.onnx"),
        read("tools/synthetic/templates/inonu-final-a.docx"),
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("sinav-kayit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    reader
        .start_recording(dir.to_string_lossy().into_owned())
        .unwrap();
    let blank = RgbImage::from_pixel(900, 1200, image::Rgb([90, 90, 90]));
    let mut saved = Vec::new();
    for index in [2, 0] {
        let sheet = image::open(repo(&format!(
            "crates/exam_infrastructure/tests/fixtures/sheets/images/syn-900-{index:05}.jpg"
        )))
        .unwrap()
        .to_rgb8();
        for _ in 0..3 {
            reader.scan_frame(frame(&blank, 0, 0, 0.0)).unwrap();
        }
        for (dx, dy) in [(0, 0), (3, -2), (-2, 4), (5, 1), (1, -3), (-4, 2)] {
            saved.extend(reader.scan_frame(frame(&sheet, dx, dy, 0.0)).unwrap().sheet);
        }
    }
    reader.stop_recording();
    assert_eq!(saved.len(), 2);
    let labels: Vec<serde_json::Value> = saved
        .iter()
        .map(|s| {
            let values: serde_json::Map<String, serde_json::Value> = s
                .cells
                .iter()
                .map(|c| (c.field.clone(), serde_json::json!(c.value)))
                .collect();
            serde_json::json!({"sheet": s.sheet_id, "confirmed": true, "values": values})
        })
        .collect();
    std::fs::write(
        dir.join("labels.json"),
        serde_json::to_vec(&labels).unwrap(),
    )
    .unwrap();

    let report = replay(&dir, &reader);
    assert_eq!(
        (
            report.app_saved,
            report.confirmed,
            report.saved,
            report.matched
        ),
        (2, 2, 2, 2),
        "{report:?}"
    );
    assert!(report.cells > 0);
    assert_eq!(report.right, report.cells, "{report:?}");
    let zip = crate::recording::pack(&dir, &[("labels.json".to_owned(), Vec::new())]).unwrap();
    assert!(zip.exists());
    let _ = std::fs::remove_file(zip);
    let _ = std::fs::remove_dir_all(&dir);
}
