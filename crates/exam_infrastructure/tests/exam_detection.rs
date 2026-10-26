//! Recognising the exam from photographed sheets alone (no Word file): the
//! result must match what the Word import says about the same exam.

use exam_core::domain::{ModelVersion, QuestionId, ScoreValue};
use exam_infrastructure::detect::{detect_exam, table_lines};
use exam_infrastructure::printed::PrintedTextReader;
use exam_infrastructure::recognition::CellRecognizer;
use exam_infrastructure::sheet::{SheetTemplate, TemplateSheetAnalyzer};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn template() -> SheetTemplate {
    SheetTemplate::load(root().join("tools/synthetic/templates/inonu-final-a.json")).unwrap()
}

fn printed() -> PrintedTextReader {
    PrintedTextReader::from_bytes(
        std::fs::read(root().join("models/ppocr-v4-rec.onnx")).unwrap(),
        &std::fs::read_to_string(root().join("models/ppocr-v4-rec.keys.txt")).unwrap(),
    )
    .unwrap()
}

fn photo(index: usize) -> image::RgbImage {
    image::open(root().join(format!(
        "crates/exam_infrastructure/tests/fixtures/sheets/images/syn-900-{index:05}.jpg"
    )))
    .unwrap()
    .to_rgb8()
}

fn word_exam() -> exam_import::ImportedExam {
    let bytes = std::fs::read(root().join("tools/synthetic/templates/inonu-final-a.docx")).unwrap();
    exam_import::ImportedExam::from_docx(&bytes).unwrap()
}

fn label(name: &str) -> serde_json::Value {
    std::fs::read_to_string(
        root().join("crates/exam_infrastructure/tests/fixtures/detect/labels.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
    .find(|v| v["image"] == format!("{name}.jpg"))
    .unwrap()
}

fn bounds(points: impl Iterator<Item = (f64, f64)>) -> [f64; 4] {
    points.fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, (x, y)| {
        [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
    })
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let inter =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |r: [f64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter)
}

fn clean(name: &str) -> image::RgbImage {
    image::open(root().join(format!(
        "crates/exam_infrastructure/tests/fixtures/detect/{name}.jpg"
    )))
    .unwrap()
    .to_rgb8()
}

/// Left edge of each column as a share of the page width.
fn column_lefts(exam: &exam_import::ImportedExam) -> Vec<f64> {
    exam.questions
        .iter()
        .map(|q| f64::from(q.score_column.left_twips) / f64::from(exam.page_width_twips))
        .collect()
}

#[test]
fn the_question_columns_are_found_on_poor_photos() {
    // Low-resolution, blurred photos with shadows: the printed texts are hard
    // to read there, but the table's columns must still be found.
    let template = template();
    let word = column_lefts(&word_exam());
    for index in 0..3 {
        let photo = photo(index);
        let located = template.locate(&photo, None);
        let lines = table_lines(&template, &photo, &located).unwrap();
        assert_eq!(lines.len(), 5, "sheet {index}: {lines:?}");
        let (page, _) = template.size();
        for (found, expected) in lines.iter().zip(&word) {
            let found = found / page as f64;
            assert!(
                (found - expected).abs() < 0.01,
                "sheet {index}: {found:.3} vs {expected:.3}"
            );
        }
    }
}

#[test]
fn the_exam_is_recognised_from_the_sheet_alone() {
    let (template, printed) = (template(), printed());
    for name in ["clean-a", "clean-b"] {
        let photo = clean(name);
        let located = template.locate(&photo, None);
        let detected = detect_exam(&template, &printed, &photo, &located).unwrap();

        let pairs: Vec<(String, Vec<String>)> = detected
            .questions
            .iter()
            .map(|q| {
                (
                    q.printed_id.clone(),
                    q.outcomes.iter().map(|o| o.as_str().to_owned()).collect(),
                )
            })
            .collect();
        assert_eq!(
            pairs,
            [("S1", "PÇ2"), ("S2", "PÇ1"), ("S3", "PÇ3"), ("S4", "PÇ2")]
                .map(|(q, o)| (q.to_owned(), vec![o.to_owned()])),
            "{name}"
        );
        let meta = &detected.metadata;
        assert_eq!(meta.course_code.as_deref(), Some("BİLM374"), "{name}");
        // The space is sometimes lost; the course is identified by its code.
        assert_eq!(
            meta.course_name.as_deref().map(|n| n.replace(' ', "")),
            Some("YAPAYZEKA".to_owned()),
            "{name}"
        );
        assert_eq!(meta.exam_kind.as_deref(), Some("Final"), "{name}");
        assert_eq!(meta.term.as_deref(), Some("Bahar"), "{name}");
        assert_eq!(meta.group.as_deref(), Some("A"), "{name}");
        // A reader laid out from the detected columns finds every table cell
        // where it is on the photo.
        let mut exam = detected.clone();
        for question in &mut exam.questions {
            question.maximum_score = Some(ScoreValue::from_decimal("100").unwrap());
        }
        let analyzer = TemplateSheetAnalyzer::for_imported_exam(
            self::template(),
            CellRecognizer::load(root().join("models/digit-crnn-v4.onnx")).unwrap(),
            &exam,
            ModelVersion::new("digit-crnn-v4", "4").unwrap(),
        )
        .unwrap();
        let sheet = analyzer.read_sheet(&photo).unwrap();
        for cell in label(name)["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            if !(key.starts_with("question:") || key == "declared_total") {
                continue;
            }
            let field = sheet.fields.iter().find(|f| f.field_key == key).unwrap();
            let expected = bounds(
                cell["quad"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())),
            );
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            assert!(overlap > 0.6, "{name} {key}: IoU {overlap:.2}");
        }
        assert!(detected.questions.iter().all(|q| q.maximum_score.is_none()));
    }
}

#[test]
fn a_reader_built_from_the_detected_exam_reads_like_the_word_one() {
    let (template, printed) = (template(), printed());
    let photo = clean("clean-a");
    let located = template.locate(&photo, None);
    let mut detected = detect_exam(&template, &printed, &photo, &located).unwrap();
    for (question, points) in detected.questions.iter_mut().zip(["20", "30", "25", "25"]) {
        question.maximum_score = Some(ScoreValue::from_decimal(points).unwrap());
    }
    let recognizer = || CellRecognizer::load(root().join("models/digit-crnn-v4.onnx")).unwrap();
    let version = || ModelVersion::new("digit-crnn-v4", "4").unwrap();
    let from_sheet =
        TemplateSheetAnalyzer::for_imported_exam(template, recognizer(), &detected, version())
            .unwrap();
    let mut word = word_exam();
    for id in ["3", "4"] {
        word = word.with_maximum_score(
            &QuestionId::new(id).unwrap(),
            ScoreValue::from_decimal("25").unwrap(),
        );
    }
    let from_word =
        TemplateSheetAnalyzer::for_imported_exam(self::template(), recognizer(), &word, version())
            .unwrap();
    let texts = |a: &TemplateSheetAnalyzer| -> Vec<String> {
        a.read_sheet(&photo)
            .unwrap()
            .fields
            .into_iter()
            .filter_map(|f| f.reading.map(|r| r.text))
            .collect()
    };
    assert_eq!(texts(&from_sheet), texts(&from_word));
}
