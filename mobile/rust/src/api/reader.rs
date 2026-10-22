//! Reading photographed sheets on the phone: the whole pipeline from photo to
//! per-PÇ results, with the cell crops the review screen shows.

use crate::api::exam::{info, ExamInfo};
use anyhow::{anyhow, Context};
use exam_application::instrumentation::NoopInstrumentation;
use exam_application::ports::analyzer::{
    AnalyzerError, DocumentPage, ExamDocumentAnalyzer, ExtractedField, RawScoreCandidate,
    StructuredPageExtraction,
};
use exam_application::roster::{RosterEntry, StudentRoster};
use exam_application::use_cases::{FinalizeExamSession, ProcessExamPage};
use exam_core::domain::{
    BoundingBox, Confidence, Exam, ExamSession, ModelVersion, OutcomeAllocationRule, QuestionId,
    ReviewIssue, ReviewReason, ScoreValue, SourceRegion, StudentNumber,
};
use exam_import::ImportedExam;
use exam_infrastructure::persistence::InMemorySessionRepository;
use exam_infrastructure::recognition::CellRecognizer;
use exam_infrastructure::sheet::{
    source_region, SheetField, SheetReading, SheetTemplate, TemplateSheetAnalyzer,
};
use exam_infrastructure::vision::{warp_quad, Point};
use std::io::Cursor;
use std::sync::Mutex;

pub struct RosterStudent {
    pub number: String,
    pub name: String,
}

/// A maximum score the instructor entered for a question, e.g. `S3` = `25`.
pub struct MaximumScore {
    pub question: String,
    pub value: String,
}

pub struct CellResult {
    /// `Numara`, `S1`… or `Not`.
    pub field: String,
    /// What the recognizer read; `~` marks a crossed-out correction.
    pub text: String,
    /// The accepted number, if the text parses as one.
    pub value: Option<String>,
    pub confidence: f32,
    /// True when the application accepted `value` (confident enough and
    /// valid); otherwise it waits for the instructor.
    pub accepted: bool,
    /// Normalised `[x, y, width, height]` of the cell in the photo.
    pub region: Vec<f32>,
    /// The cell cut out of the photo (PNG), for the review screen; empty
    /// for a confirmed sheet.
    pub crop_png: Vec<u8>,
}

pub struct IssueResult {
    /// Stable key, e.g. `low_confidence`, `total_mismatch`, `roster_near_match`.
    pub reason: String,
    pub field: String,
    pub detail: Option<String>,
}

pub struct OutcomeResult {
    pub id: String,
    /// Points from the accepted scores of the questions mapped to this PÇ.
    pub earned: String,
    pub possible: String,
    pub percent: f64,
    /// False while some question of this PÇ has no accepted score yet; the
    /// numbers then cover only the accepted questions.
    pub complete: bool,
}

pub struct SheetResult {
    pub sheet_id: String,
    /// False when the printed form was not recognised convincingly.
    pub page_found: bool,
    pub student_number: Option<String>,
    pub student_name: Option<String>,
    pub cells: Vec<CellResult>,
    pub issues: Vec<IssueResult>,
    pub outcomes: Vec<OutcomeResult>,
    pub requires_review: bool,
}

struct Precomputed(StructuredPageExtraction);

impl ExamDocumentAnalyzer for Precomputed {
    fn analyze(&self, _: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        Ok(self.0.clone())
    }
}

#[flutter_rust_bridge::frb(opaque)]
pub struct ExamReader {
    pub(crate) analyzer: TemplateSheetAnalyzer,
    imported: ImportedExam,
    exam: Exam,
    roster: Option<StudentRoster>,
    sessions: Mutex<InMemorySessionRepository>,
    pub(crate) live: Mutex<crate::live::LiveState>,
}

fn reason_key(reason: &ReviewReason) -> &'static str {
    match reason {
        ReviewReason::LowConfidence => "low_confidence",
        ReviewReason::ScoreExceedsMaximum => "score_exceeds_maximum",
        ReviewReason::UnknownQuestion => "unknown_question",
        ReviewReason::UnresolvedStudentIdentity => "unresolved_student_identity",
        ReviewReason::AmbiguousOutcomeMapping => "ambiguous_outcome_mapping",
        ReviewReason::AmbiguousCorrectedScore => "ambiguous_corrected_score",
        ReviewReason::InconsistentPageIdentity => "inconsistent_page_identity",
        ReviewReason::SourceRegionMismatch => "source_region_mismatch",
        ReviewReason::ConflictingQuestionScore => "conflicting_question_score",
        ReviewReason::InvalidCandidate => "invalid_candidate",
        ReviewReason::TotalMismatch => "total_mismatch",
        ReviewReason::BlankField => "blank_field",
        ReviewReason::RosterNearMatch => "roster_near_match",
    }
}

fn png(image: &image::RgbImage) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    Ok(bytes)
}

fn percent(earned: ScoreValue, possible: ScoreValue) -> f64 {
    if possible.micro_points() == 0 {
        0.0
    } else {
        100.0 * earned.micro_points() as f64 / possible.micro_points() as f64
    }
}

impl ExamReader {
    /// Builds a reader for one exam. `template_*` and `model_onnx` come from
    /// the app's assets, `exam_docx` from the instructor.
    pub fn new(
        template_json: String,
        template_png: Vec<u8>,
        model_onnx: Vec<u8>,
        exam_docx: Vec<u8>,
        maximum_scores: Vec<MaximumScore>,
        roster: Vec<RosterStudent>,
    ) -> anyhow::Result<ExamReader> {
        let mut imported = ImportedExam::from_docx(&exam_docx)?;
        for entry in &maximum_scores {
            let id = QuestionId::new(&entry.question).map_err(|e| anyhow!("{e}"))?;
            let value = ScoreValue::from_decimal(&entry.value.replace(',', "."))
                .map_err(|e| anyhow!("{}: {e}", entry.question))?;
            imported = imported.with_maximum_score(&id, value);
        }
        let exam_id = imported
            .metadata
            .course_code
            .clone()
            .unwrap_or_else(|| "exam".to_owned());
        let exam = imported
            .to_exam(
                &exam_id,
                imported.metadata.course.as_deref().unwrap_or("Sınav"),
            )
            .map_err(|e| anyhow!("{e}"))?;
        let template =
            SheetTemplate::from_bytes(&template_json, &template_png).map_err(|e| anyhow!("{e}"))?;
        let recognizer = CellRecognizer::from_bytes(&model_onnx).map_err(|e| anyhow!("{e}"))?;
        let analyzer = TemplateSheetAnalyzer::for_imported_exam(
            template,
            recognizer,
            &imported,
            ModelVersion::new("digit-crnn-v4", "4").map_err(|e| anyhow!("{e}"))?,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let roster = if roster.is_empty() {
            None
        } else {
            Some(
                StudentRoster::new(
                    roster
                        .into_iter()
                        .map(|s| {
                            Ok(RosterEntry {
                                number: StudentNumber::new(s.number).map_err(|e| anyhow!("{e}"))?,
                                name: s.name,
                            })
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?,
                )
                .map_err(|e| anyhow!("{e}"))?,
            )
        };
        Ok(ExamReader {
            analyzer,
            imported,
            exam,
            roster,
            sessions: Mutex::new(InMemorySessionRepository::default()),
            live: Mutex::new(crate::live::LiveState::default()),
        })
    }

    #[flutter_rust_bridge::frb(sync)]
    pub fn exam_info(&self) -> ExamInfo {
        info(&self.imported)
    }

    /// Reads one photographed sheet. Scanning the same `sheet_id` again
    /// replaces the earlier reading.
    pub fn process_photo(&self, sheet_id: String, photo: Vec<u8>) -> anyhow::Result<SheetResult> {
        let rgb = image::load_from_memory(&photo)
            .context("fotoğraf okunamadı")?
            .to_rgb8();
        let sheet = self.analyzer.read_sheet(&rgb).map_err(|e| anyhow!("{e}"))?;
        self.sheet_result(sheet_id, &rgb, &sheet, true)
    }

    /// The result for a sheet read from `rgb` (a photo or a camera frame),
    /// validated by the application. Cell crops are cut out only when
    /// `with_crops` is set, since encoding them takes time.
    pub(crate) fn sheet_result(
        &self,
        sheet_id: String,
        rgb: &image::RgbImage,
        sheet: &SheetReading,
        with_crops: bool,
    ) -> anyhow::Result<SheetResult> {
        let extraction = self.analyzer.extraction(0, sheet);

        let mut cells = Vec::new();
        for field in &sheet.fields {
            let Some(reading) = &field.reading else {
                continue;
            };
            let label = match field.field {
                SheetField::StudentNumber => NUMBER_FIELD.to_owned(),
                SheetField::Score(index) => extraction.questions[index]
                    .id
                    .raw_text
                    .clone()
                    .unwrap_or_default(),
                SheetField::DeclaredTotal => TOTAL_FIELD.to_owned(),
                SheetField::StudentName => continue,
            };
            let value = match field.field {
                SheetField::Score(index) => extraction.questions[index]
                    .score
                    .as_ref()
                    .and_then(|s| s.earned.candidate)
                    .map(|v| v.to_string()),
                SheetField::DeclaredTotal => extraction
                    .declared_total
                    .as_ref()
                    .and_then(|t| t.candidate)
                    .map(|v| v.to_string()),
                _ => (!reading.is_blank()).then(|| reading.value_text()),
            };
            let top = (field.quad[1].x - field.quad[0].x).hypot(field.quad[1].y - field.quad[0].y);
            let side = (field.quad[3].x - field.quad[0].x).hypot(field.quad[3].y - field.quad[0].y);
            let width = ((96.0 * top / side.max(1.0)).round() as u32).clamp(16, 1200);
            let region = source_region(0, &field.quad, sheet.width, sheet.height)
                .map(|r| vec![r.bounds.x, r.bounds.y, r.bounds.width, r.bounds.height])
                .unwrap_or_default();
            cells.push(CellResult {
                field: label,
                text: reading.text.clone(),
                value,
                confidence: reading.confidence,
                accepted: false,
                region,
                crop_png: if with_crops {
                    png(&warp_quad(rgb, &field.quad, width, 96))?
                } else {
                    Vec::new()
                },
            });
        }
        self.evaluate(sheet_id, sheet.page_found, &extraction, cells)
    }

    /// Reads one frame of the live camera preview. Call it for frame after
    /// frame (skipping frames while a call is running): it follows the sheet,
    /// combines the readings of consecutive frames and reports `Saved` with the
    /// result once, as soon as the sheet is read reliably. The next sheet is
    /// read when it comes into view.
    pub fn scan_frame(
        &self,
        frame: crate::api::camera::CameraFrame,
    ) -> anyhow::Result<crate::api::camera::ScanUpdate> {
        crate::live::scan(self, frame)
    }

    /// Forgets the sheet being scanned and the last saved one, e.g. when the
    /// scanning screen is opened again.
    #[flutter_rust_bridge::frb(sync)]
    pub fn reset_scan(&self) {
        if let Ok(mut state) = self.live.lock() {
            *state = crate::live::LiveState::default();
        }
    }

    /// Re-evaluates a sheet with the values the instructor checked or
    /// corrected. They count as verified (full confidence); validation still
    /// applies, so e.g. a total that does not add up is reported again.
    /// `values` carries every field of the sheet with the region it was read
    /// from; crops are not returned again.
    pub fn confirm_sheet(
        &self,
        sheet_id: String,
        values: Vec<FieldValue>,
    ) -> anyhow::Result<SheetResult> {
        let blank = SheetReading {
            width: 1,
            height: 1,
            page_found: true,
            registration_score: 1.0,
            page_quad: [Point::new(0.0, 0.0); 4],
            fields: Vec::new(),
        };
        let mut extraction = self.analyzer.extraction(0, &blank);
        let mut cells = Vec::new();
        for entry in &values {
            let value = entry
                .value
                .as_deref()
                .map(|v| v.trim().replace(',', "."))
                .filter(|v| !v.is_empty());
            let region = region_from(&entry.region);
            let field = |candidate| verified(candidate, value.clone(), region.clone());
            let score_field = |v: &String| {
                verified(
                    ScoreValue::from_decimal(v).ok(),
                    value.clone(),
                    region.clone(),
                )
            };
            if entry.field == NUMBER_FIELD {
                extraction
                    .student
                    .as_mut()
                    .expect("student field")
                    .student_number = field(value.clone());
            } else if entry.field == TOTAL_FIELD {
                extraction.declared_total = value.as_ref().map(score_field);
            } else {
                let id = QuestionId::new(&entry.field).map_err(|e| anyhow!("{e}"))?;
                let question = extraction
                    .questions
                    .iter_mut()
                    .find(|q| q.id.candidate.as_ref() == Some(&id))
                    .ok_or_else(|| anyhow!("{} bu sınavda yok", entry.field))?;
                question.score = value.as_ref().map(|v| RawScoreCandidate {
                    earned: score_field(v),
                    maximum: None,
                });
            }
            if value.is_none() {
                extraction.review_issues.push(ReviewIssue::new(
                    ReviewReason::BlankField,
                    if entry.field == NUMBER_FIELD {
                        "student number".to_owned()
                    } else if entry.field == TOTAL_FIELD {
                        "declared total".to_owned()
                    } else {
                        format!(
                            "question {} score",
                            QuestionId::new(&entry.field).map_err(|e| anyhow!("{e}"))?
                        )
                    },
                    None,
                    region.clone(),
                ));
            }
            cells.push(CellResult {
                field: entry.field.clone(),
                text: value.clone().unwrap_or_default(),
                value,
                confidence: 1.0,
                accepted: false,
                region: entry.region.clone(),
                crop_png: Vec::new(),
            });
        }
        self.evaluate(sheet_id, true, &extraction, cells)
    }

    /// Runs the application use cases on an extraction and reports the result.
    fn evaluate(
        &self,
        sheet_id: String,
        page_found: bool,
        extraction: &StructuredPageExtraction,
        mut cells: Vec<CellResult>,
    ) -> anyhow::Result<SheetResult> {
        let mut repository = self
            .sessions
            .lock()
            .map_err(|_| anyhow!("session store is poisoned"))?;
        let mut session = ExamSession::new(&sheet_id, Some(1))
            .map_err(|e| anyhow!("{e}"))?
            .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight);
        session
            .merge_exam_discoveries(self.exam.clone())
            .map_err(|e| anyhow!("{e}"))?;
        repository.insert(session);

        let precomputed = Precomputed(extraction.clone());
        let instrumentation = NoopInstrumentation;
        let mut use_case = ProcessExamPage::new(&precomputed, &mut *repository, &instrumentation);
        if let Some(roster) = &self.roster {
            use_case = use_case.with_roster(roster);
        }
        let page = use_case.execute(&sheet_id, DocumentPage::new(0, vec![1]));
        let finalized = FinalizeExamSession::new(&*repository)
            .execute(&sheet_id)
            .map_err(|e| anyhow!("{e}"))?;
        let session = repository
            .get(&sheet_id)
            .ok_or_else(|| anyhow!("session disappeared"))?;

        let parse =
            |v: &Option<String>| v.as_deref().and_then(|v| ScoreValue::from_decimal(v).ok());
        for cell in &mut cells {
            cell.accepted = if cell.field == NUMBER_FIELD {
                session.student().map(|s| s.number.as_str()) == cell.value.as_deref()
            } else if cell.field == TOTAL_FIELD {
                page.as_ref()
                    .ok()
                    .and_then(|p| p.declared_total.as_ref())
                    .map(|t| t.value)
                    == parse(&cell.value)
            } else {
                QuestionId::new(&cell.field).ok().is_some_and(|id| {
                    session
                        .canonical_question_results()
                        .get(&id)
                        .and_then(|r| r.score)
                        .map(|s| s.value())
                        == parse(&cell.value)
                }) && cell.value.is_some()
            };
        }

        let mut issues: Vec<IssueResult> = match &page {
            Ok(page) => page
                .review_issues
                .iter()
                .map(|issue| IssueResult {
                    reason: reason_key(&issue.reason).to_owned(),
                    field: issue.field.clone(),
                    detail: issue.candidate.clone(),
                })
                .collect(),
            Err(error) => vec![IssueResult {
                reason: "page_failed".to_owned(),
                field: "page".to_owned(),
                detail: Some(error.to_string()),
            }],
        };
        if session.student().is_none()
            && !issues
                .iter()
                .any(|i| i.reason == "unresolved_student_identity")
        {
            issues.push(IssueResult {
                reason: "unresolved_student_identity".to_owned(),
                field: "student identity".to_owned(),
                detail: None,
            });
        }

        Ok(SheetResult {
            sheet_id,
            page_found,
            student_number: session.student().map(|s| s.number.as_str().to_owned()),
            student_name: session.student().map(|s| s.name.clone()),
            cells,
            issues,
            // Every PÇ of the exam is listed, also those still waiting for a
            // score, so the instructor sees what is missing.
            outcomes: self
                .exam
                .learning_outcomes
                .keys()
                .map(|id| {
                    let accepted = finalized.learning_outcomes.get(id);
                    let (earned, possible) = accepted
                        .map(|o| (o.earned_score, o.possible_score))
                        .unwrap_or((ScoreValue::ZERO, ScoreValue::ZERO));
                    let complete = self
                        .exam
                        .questions
                        .values()
                        .filter(|q| q.outcome_allocations.iter().any(|a| &a.outcome_id == id))
                        .all(|q| {
                            session
                                .canonical_question_results()
                                .get(&q.id)
                                .is_some_and(|r| r.score.is_some())
                        });
                    OutcomeResult {
                        id: id.as_str().to_owned(),
                        earned: earned.to_string(),
                        possible: possible.to_string(),
                        percent: percent(earned, possible),
                        complete,
                    }
                })
                .collect(),
            requires_review: finalized.requires_review,
        })
    }

    /// An Excel workbook for the exam: one row per sheet, the class result per
    /// PÇ and per question, and the exam details.
    pub fn export_excel(&self, sheets: Vec<ReportSheet>) -> anyhow::Result<Vec<u8>> {
        crate::report::workbook(&self.imported, &self.exam, &sheets)
    }
}

/// A checked value of one field, e.g. `S2` = `17.5`, with the normalised
/// region `[x, y, width, height]` it was read from.
pub struct FieldValue {
    pub field: String,
    pub value: Option<String>,
    pub region: Vec<f32>,
}

/// One saved sheet as it goes into the Excel report.
pub struct ReportSheet {
    pub student_number: Option<String>,
    pub student_name: Option<String>,
    /// Final value per field (`S1`…, `Not`); for an unconfirmed sheet pass
    /// only the values the application accepted.
    pub values: Vec<FieldValue>,
    pub outcomes: Vec<OutcomeResult>,
    /// True once the instructor confirmed the sheet.
    pub confirmed: bool,
}

pub(crate) const NUMBER_FIELD: &str = "Numara";
pub(crate) const TOTAL_FIELD: &str = "Not";

/// A field value the instructor checked: full confidence.
fn verified<T>(
    candidate: Option<T>,
    raw: Option<String>,
    region: Option<SourceRegion>,
) -> ExtractedField<T> {
    ExtractedField::new(candidate, raw, Confidence::new(1.0).ok(), region)
}

fn region_from(values: &[f32]) -> Option<SourceRegion> {
    let [x, y, width, height] = values else {
        return None;
    };
    BoundingBox::new(*x, *y, *width, *height)
        .ok()
        .map(|bounds| SourceRegion::new(0, bounds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(path: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    }

    #[test]
    fn a_photo_goes_through_the_bridge_to_outcome_results() {
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        let labels = std::fs::read_to_string(repo(
            "crates/exam_infrastructure/tests/fixtures/sheets/labels.jsonl",
        ))
        .unwrap();
        let first = labels.lines().next().unwrap();
        let number = first
            .split("\"number\": \"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let reader = ExamReader::new(
            String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
            read("tools/synthetic/templates/inonu-final-a.png"),
            read("models/digit-crnn-v4.onnx"),
            read("tools/synthetic/templates/inonu-final-a.docx"),
            vec![
                MaximumScore {
                    question: "S3".into(),
                    value: "25".into(),
                },
                MaximumScore {
                    question: "S4".into(),
                    value: "25".into(),
                },
            ],
            vec![RosterStudent {
                number: number.to_owned(),
                name: "Test Öğrenci".into(),
            }],
        )
        .unwrap();
        assert!(reader.exam_info().ready);
        let result = reader
            .process_photo(
                "sheet-1".into(),
                read("crates/exam_infrastructure/tests/fixtures/sheets/images/syn-900-00000.jpg"),
            )
            .unwrap();
        assert!(result.page_found);
        assert_eq!(result.cells.len(), 6); // number, four scores, total
        assert!(result
            .cells
            .iter()
            .all(|c| c.crop_png.starts_with(b"\x89PNG")));
        let outcomes: Vec<&str> = result.outcomes.iter().map(|o| o.id.as_str()).collect();
        assert!(!outcomes.is_empty() && outcomes.iter().all(|o| ["PÇ1", "PÇ2", "PÇ3"].contains(o)));
        assert!(result
            .outcomes
            .iter()
            .all(|o| (0.0..=100.0).contains(&o.percent)));
    }

    fn sample_reader(number: &str) -> ExamReader {
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        ExamReader::new(
            String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
            read("tools/synthetic/templates/inonu-final-a.png"),
            read("models/digit-crnn-v4.onnx"),
            read("tools/synthetic/templates/inonu-final-a.docx"),
            vec![
                MaximumScore {
                    question: "S3".into(),
                    value: "25".into(),
                },
                MaximumScore {
                    question: "S4".into(),
                    value: "25".into(),
                },
            ],
            vec![RosterStudent {
                number: number.into(),
                name: "Zeynep Demir".into(),
            }],
        )
        .unwrap()
    }

    fn confirmed(result: &SheetResult, overrides: &[(&str, &str)]) -> Vec<FieldValue> {
        result
            .cells
            .iter()
            .map(|cell| FieldValue {
                field: cell.field.clone(),
                value: overrides
                    .iter()
                    .find(|(f, _)| *f == cell.field)
                    .map(|(_, v)| (*v).to_owned())
                    .or_else(|| cell.value.clone()),
                region: cell.region.clone(),
            })
            .collect()
    }

    #[test]
    fn corrected_values_become_final_results() {
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        let reader = sample_reader("04451095728");
        let photo =
            read("crates/exam_infrastructure/tests/fixtures/sheets/images/syn-900-00001.jpg");
        let first = reader.process_photo("s".into(), photo).unwrap();
        assert!(first.cells.iter().all(|c| c.region.len() == 4));
        // The true values of this synthetic sheet.
        let truth = [
            ("Numara", "04451095728"),
            ("S1", "13"),
            ("S2", "23"),
            ("S3", "23"),
            ("S4", "11"),
            ("Not", "70"),
        ];
        let result = reader
            .confirm_sheet("s".into(), confirmed(&first, &truth))
            .unwrap();
        assert!(
            result.issues.is_empty(),
            "{:?}",
            result.issues.iter().map(|i| &i.reason).collect::<Vec<_>>()
        );
        assert!(!result.requires_review);
        assert_eq!(result.student_name.as_deref(), Some("Zeynep Demir"));
        let pc1 = result.outcomes.iter().find(|o| o.id == "PÇ1").unwrap();
        assert!(pc1.complete);
        assert_eq!((pc1.earned.as_str(), pc1.possible.as_str()), ("23", "30"));
        assert!(result.outcomes.iter().all(|o| o.complete));
        assert!(result
            .cells
            .iter()
            .all(|c| c.confidence == 1.0 && c.crop_png.is_empty()));

        // The instructor's own arithmetic on paper is still checked.
        let wrong_total = reader
            .confirm_sheet(
                "s".into(),
                confirmed(&first, &[("S4", "11"), ("Not", "75")]),
            )
            .unwrap();
        assert!(wrong_total
            .issues
            .iter()
            .any(|i| i.reason == "total_mismatch"));
        assert!(wrong_total.requires_review);

        // A score above the question maximum is refused.
        let too_high = reader
            .confirm_sheet("s".into(), confirmed(&first, &[("S1", "21")]))
            .unwrap();
        assert!(too_high
            .issues
            .iter()
            .any(|i| i.reason == "score_exceeds_maximum"));

        let report = reader
            .export_excel(vec![ReportSheet {
                student_number: result.student_number.clone(),
                student_name: result.student_name.clone(),
                values: confirmed(&result, &[]),
                outcomes: result.outcomes,
                confirmed: true,
            }])
            .unwrap();
        assert!(report.starts_with(b"PK"), "xlsx files are zip archives");
        assert!(report.len() > 4_000);
    }

    /// Writes a sample report for manual inspection:
    /// `SAMPLE_REPORT=out.xlsx cargo test -p rust_lib_sinav_okuyucu -- --ignored`.
    #[test]
    #[ignore]
    fn write_sample_report() {
        let Ok(path) = std::env::var("SAMPLE_REPORT") else {
            return;
        };
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        let reader = sample_reader("04451095728");
        let mut sheets = Vec::new();
        for (i, name) in ["syn-900-00000", "syn-900-00001", "syn-900-00002"]
            .iter()
            .enumerate()
        {
            let photo = read(&format!(
                "crates/exam_infrastructure/tests/fixtures/sheets/images/{name}.jpg"
            ));
            let result = reader.process_photo(format!("s{i}"), photo).unwrap();
            sheets.push(ReportSheet {
                student_number: result.student_number.clone(),
                student_name: result.student_name.clone(),
                values: result
                    .cells
                    .iter()
                    .map(|c| FieldValue {
                        field: c.field.clone(),
                        value: c.accepted.then(|| c.value.clone()).flatten(),
                        region: c.region.clone(),
                    })
                    .collect(),
                outcomes: result.outcomes,
                confirmed: false,
            });
        }
        std::fs::write(path, reader.export_excel(sheets).unwrap()).unwrap();
    }

    #[test]
    fn an_exam_with_missing_points_is_refused() {
        let read = |p: &str| std::fs::read(repo(p)).unwrap();
        let result = ExamReader::new(
            String::from_utf8(read("tools/synthetic/templates/inonu-final-a.json")).unwrap(),
            read("tools/synthetic/templates/inonu-final-a.png"),
            read("models/digit-crnn-v4.onnx"),
            read("tools/synthetic/templates/inonu-final-a.docx"),
            Vec::new(),
            Vec::new(),
        );
        assert!(result.is_err());
    }
}
