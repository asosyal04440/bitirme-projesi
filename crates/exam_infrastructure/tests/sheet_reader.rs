//! End-to-end reading of synthetic phone photos of the İnönü sheet
//! (`tools/synthetic/generate.py --seed 900 --long-side 1200 --mnist-split test`).

use exam_application::instrumentation::NoopInstrumentation;
use exam_application::ports::analyzer::{DocumentPage, ExamDocumentAnalyzer};
use exam_application::roster::{RosterEntry, StudentRoster};
use exam_application::use_cases::{FinalizeExamSession, ProcessExamPage};
use exam_core::domain::{
    ExamSession, LearningOutcomeId, ModelVersion, OutcomeAllocationRule, QuestionId, ScoreValue,
    StudentNumber,
};
use exam_infrastructure::persistence::InMemorySessionRepository;
use exam_infrastructure::recognition::CellRecognizer;
use exam_infrastructure::sheet::{SheetTemplate, TemplateSheetAnalyzer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> PathBuf {
    root().join("tests/fixtures/sheets")
}

fn labels() -> Vec<Value> {
    std::fs::read_to_string(fixtures().join("labels.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn analyzer() -> TemplateSheetAnalyzer {
    let template =
        SheetTemplate::load(root().join("../../tools/synthetic/templates/inonu-final-a.json"))
            .unwrap();
    let recognizer = CellRecognizer::load(root().join("../../models/digit-crnn-v4.onnx")).unwrap();
    let maximum_scores: BTreeMap<QuestionId, ScoreValue> =
        [("S1", "20"), ("S2", "30"), ("S3", "25"), ("S4", "25")]
            .into_iter()
            .map(|(q, m)| {
                (
                    QuestionId::new(q).unwrap(),
                    ScoreValue::from_decimal(m).unwrap(),
                )
            })
            .collect();
    TemplateSheetAnalyzer::new(
        template,
        recognizer,
        maximum_scores,
        ModelVersion::new("digit-crnn-v4", "4").unwrap(),
    )
    .unwrap()
}

fn photo_bytes(label: &Value) -> Vec<u8> {
    std::fs::read(fixtures().join(label["image"].as_str().unwrap())).unwrap()
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

#[test]
fn every_field_is_located_on_the_photographed_sheet() {
    let analyzer = analyzer();
    for label in labels() {
        let photo = image::load_from_memory(&photo_bytes(&label))
            .unwrap()
            .to_rgb8();
        let sheet = analyzer.read_sheet(&photo).unwrap();
        assert!(
            sheet.page_found,
            "{}: {}",
            label["sample_id"], sheet.registration_score
        );
        for cell in label["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            let field = sheet.fields.iter().find(|f| f.field_key == key).unwrap();
            let expected = bounds(
                cell["quad"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())),
            );
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            assert!(
                overlap > 0.6,
                "{} {key}: IoU {overlap:.2}",
                label["sample_id"]
            );
        }
    }
}

#[test]
fn scores_and_structure_are_extracted_through_the_analyzer_port() {
    let analyzer = analyzer();
    let (mut correct, mut total) = (0, 0);
    for label in labels() {
        let extraction = analyzer
            .analyze(&DocumentPage::new(0, photo_bytes(&label)))
            .unwrap();
        assert_eq!(extraction.questions.len(), 4);
        assert_eq!(
            extraction
                .outcome_mappings
                .iter()
                .map(|m| (
                    m.question_id.candidate.clone().unwrap(),
                    m.outcome_id.candidate.clone().unwrap()
                ))
                .collect::<Vec<_>>(),
            [("1", "PÇ2"), ("2", "PÇ1"), ("3", "PÇ3"), ("4", "PÇ2")].map(|(q, o)| (
                QuestionId::new(q).unwrap(),
                LearningOutcomeId::new(o).unwrap()
            ))
        );
        for cell in label["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            let Some(index) = key
                .strip_prefix("question:")
                .and_then(|rest| rest.strip_suffix(".earned_score"))
                .and_then(|id| id.parse::<usize>().ok())
            else {
                continue;
            };
            let expected = cell["value"]
                .as_str()
                .map(|v| ScoreValue::from_decimal(v).unwrap());
            let earned = extraction.questions[index - 1]
                .score
                .as_ref()
                .map(|s| &s.earned);
            let actual = earned.and_then(|e| e.candidate);
            total += 1;
            if actual == expected {
                correct += 1;
            } else {
                // A misread is acceptable only if it is sent to review.
                let confidence = earned.and_then(|e| e.confidence).map_or(0.0, |c| c.value());
                assert!(
                    confidence < 0.8,
                    "{} {key}: read {actual:?} instead of {expected:?} with confidence {confidence}",
                    label["sample_id"]
                );
            }
        }
    }
    assert!(
        correct * 3 >= total * 2,
        "{correct}/{total} scores read correctly"
    );
}

#[test]
fn photo_to_learning_outcome_results_through_the_use_cases() {
    let analyzer = analyzer();
    let labels = labels();
    let roster = StudentRoster::new(labels.iter().map(|l| RosterEntry {
        number: StudentNumber::new(l["student"]["number"].as_str().unwrap()).unwrap(),
        name: l["student"]["name"].as_str().unwrap().to_owned(),
    }))
    .unwrap();
    let mut resolved = 0;
    for label in &labels {
        let id = label["sample_id"].as_str().unwrap();
        let mut repository = InMemorySessionRepository::default();
        repository.insert(
            ExamSession::new(id, Some(1))
                .unwrap()
                .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight),
        );
        let instrumentation = NoopInstrumentation;
        ProcessExamPage::new(&analyzer, &mut repository, &instrumentation)
            .with_roster(&roster)
            .execute(id, DocumentPage::new(0, photo_bytes(label)))
            .unwrap();
        let finalized = FinalizeExamSession::new(&repository).execute(id).unwrap();
        // Every question maps to one PÇ, so all three outcomes are aggregated
        // from whatever scores were accepted.
        let outcomes: Vec<&str> = finalized
            .learning_outcomes
            .keys()
            .map(LearningOutcomeId::as_str)
            .collect();
        assert!(!outcomes.is_empty(), "{id}: no outcome results");
        assert!(outcomes.iter().all(|o| ["PÇ1", "PÇ2", "PÇ3"].contains(o)));
        let session = repository.get(id).unwrap();
        if let Some(student) = session.student() {
            assert_eq!(
                student.number.as_str(),
                label["student"]["number"].as_str().unwrap(),
                "{id}: wrong student"
            );
            resolved += 1;
        }
    }
    assert!(resolved >= 2, "only {resolved}/3 students resolved");
}

fn imported_exam() -> exam_import::ImportedExam {
    let bytes =
        std::fs::read(root().join("../../tools/synthetic/templates/inonu-final-a.docx")).unwrap();
    let twenty_five = ScoreValue::from_decimal("25").unwrap();
    exam_import::ImportedExam::from_docx(&bytes)
        .unwrap()
        .with_maximum_score(&QuestionId::new("S3").unwrap(), twenty_five)
        .with_maximum_score(&QuestionId::new("S4").unwrap(), twenty_five)
}

#[test]
fn a_reader_built_from_the_word_file_finds_the_same_cells() {
    let analyzer = TemplateSheetAnalyzer::for_imported_exam(
        SheetTemplate::load(root().join("../../tools/synthetic/templates/inonu-final-a.json"))
            .unwrap(),
        CellRecognizer::load(root().join("../../models/digit-crnn-v4.onnx")).unwrap(),
        &imported_exam(),
        ModelVersion::new("digit-crnn-v4", "4").unwrap(),
    )
    .unwrap();
    for label in labels() {
        let photo = image::load_from_memory(&photo_bytes(&label))
            .unwrap()
            .to_rgb8();
        let sheet = analyzer.read_sheet(&photo).unwrap();
        for cell in label["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            let field = sheet.fields.iter().find(|f| f.field_key == key).unwrap();
            let expected = bounds(
                cell["quad"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())),
            );
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            assert!(
                overlap > 0.6,
                "{} {key}: IoU {overlap:.2}",
                label["sample_id"]
            );
        }
        let extraction = analyzer.extraction(0, &sheet);
        let maxima: Vec<String> = extraction
            .questions
            .iter()
            .map(|q| q.maximum_score.candidate.unwrap().to_string())
            .collect();
        assert_eq!(maxima, ["20", "30", "25", "25"]);
    }
}

#[test]
fn an_incomplete_import_cannot_build_a_reader() {
    let bytes =
        std::fs::read(root().join("../../tools/synthetic/templates/inonu-final-a.docx")).unwrap();
    let missing_points = exam_import::ImportedExam::from_docx(&bytes).unwrap();
    let result = TemplateSheetAnalyzer::for_imported_exam(
        SheetTemplate::load(root().join("../../tools/synthetic/templates/inonu-final-a.json"))
            .unwrap(),
        CellRecognizer::load(root().join("../../models/digit-crnn-v4.onnx")).unwrap(),
        &missing_points,
        ModelVersion::new("digit-crnn-v4", "4").unwrap(),
    );
    assert!(result.is_err());
}

/// The photo moved by `(dx, dy)` pixels, as the next camera frame of a hand-held phone.
fn shifted(photo: &image::RgbImage, dx: i64, dy: i64) -> image::RgbImage {
    let mut frame =
        image::RgbImage::from_pixel(photo.width(), photo.height(), image::Rgb([90, 90, 90]));
    image::imageops::overlay(&mut frame, photo, dx, dy);
    frame
}

fn score_texts(sheet: &exam_infrastructure::sheet::SheetReading) -> Vec<String> {
    sheet
        .fields
        .iter()
        .filter(|f| f.field_key.starts_with("question:") || f.field_key == "declared_total")
        .map(|f| f.reading.as_ref().unwrap().text.clone())
        .collect()
}

#[test]
fn a_moving_sheet_is_tracked_between_camera_frames() {
    let analyzer = analyzer();
    for label in labels() {
        let photo = image::load_from_memory(&photo_bytes(&label))
            .unwrap()
            .to_rgb8();
        let first = analyzer.locate(&photo, None);
        assert!(first.found() && !first.tracked, "{}", label["sample_id"]);

        let next = shifted(&photo, 14, -9);
        let tracked = analyzer.locate(&next, Some(&first.page_quad));
        assert!(
            tracked.found() && tracked.tracked,
            "{}: {}",
            label["sample_id"],
            tracked.registration_score
        );
        let before = analyzer.read_at(&photo, &first).unwrap();
        let after = analyzer.read_at(&next, &tracked).unwrap();
        // Every cell is still found where it now is (only the matched top of
        // the page is constrained, so the page corners themselves may drift).
        for cell in label["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            let field = after.fields.iter().find(|f| f.field_key == key).unwrap();
            let expected = bounds(
                cell["quad"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap() + 14.0, p[1].as_f64().unwrap() - 9.0)),
            );
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            assert!(
                overlap > 0.6,
                "{} {key}: IoU {overlap:.2}",
                label["sample_id"]
            );
        }
        assert_eq!(score_texts(&after), score_texts(&before));
    }
}

#[test]
fn tracking_falls_back_to_a_full_search_when_the_sheet_is_elsewhere() {
    let analyzer = analyzer();
    let label = &labels()[0];
    let photo = image::load_from_memory(&photo_bytes(label))
        .unwrap()
        .to_rgb8();
    let far_away = [
        exam_infrastructure::vision::Point::new(0.0, 0.0),
        exam_infrastructure::vision::Point::new(40.0, 0.0),
        exam_infrastructure::vision::Point::new(40.0, 50.0),
        exam_infrastructure::vision::Point::new(0.0, 50.0),
    ];
    let located = analyzer.locate(&photo, Some(&far_away));
    assert!(located.found() && !located.tracked);
}

#[test]
fn a_close_up_of_the_top_of_the_sheet_is_read() {
    let analyzer = analyzer();
    for label in labels() {
        let photo = image::load_from_memory(&photo_bytes(&label))
            .unwrap()
            .to_rgb8();
        // Only the header and the Soru-PÇ table, held closer to the camera.
        let page = analyzer.locate(&photo, None).page_quad;
        let top = page.iter().map(|p| p.y).fold(f64::MAX, f64::min).max(0.0);
        let bottom = top + 0.45 * (page[3].y - page[0].y);
        let (y0, height) = (top as u32, (bottom - top) as u32);
        let crop = image::imageops::crop_imm(&photo, 0, y0, photo.width(), height).to_image();
        let scale = 1.5;
        let close_up = image::imageops::resize(
            &crop,
            (f64::from(crop.width()) * scale) as u32,
            (f64::from(crop.height()) * scale) as u32,
            image::imageops::FilterType::Triangle,
        );

        let sheet = analyzer.read_sheet(&close_up).unwrap();
        assert!(
            sheet.page_found,
            "{}: {}",
            label["sample_id"], sheet.registration_score
        );
        for cell in label["cells"].as_array().unwrap() {
            let key = cell["field_key"].as_str().unwrap();
            if !(key.starts_with("question:") || key == "declared_total") {
                continue;
            }
            let field = sheet.fields.iter().find(|f| f.field_key == key).unwrap();
            let expected = bounds(cell["quad"].as_array().unwrap().iter().map(|p| {
                (
                    p[0].as_f64().unwrap() * scale,
                    (p[1].as_f64().unwrap() - f64::from(y0)) * scale,
                )
            }));
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            assert!(
                overlap > 0.55,
                "{} {key}: IoU {overlap:.2}",
                label["sample_id"]
            );
        }
    }
}
