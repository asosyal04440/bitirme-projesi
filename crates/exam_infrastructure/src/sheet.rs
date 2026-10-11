//! Reads a photographed exam sheet whose layout is known from its template
//! (`tools/synthetic/templates/*.json`, extracted from the exam's Word file).
//!
//! The printed structure (questions, their PÇ, the cells) comes from the
//! template; only handwriting is recognised: the student number, one score
//! per question and the `Not:` total. Names are never read; the application
//! takes them from the course roster.

use crate::recognition::{CellKind, CellReading, CellRecognizer, RgbImage as CellImage};
use crate::vision::{detect_page, warp_quad, Aligner, Homography, Point, Rect, Registrar};
use exam_application::ports::analyzer::{
    AnalyzerError, DocumentPage, ExamDocumentAnalyzer, ExtractedField, ExtractedLearningOutcome,
    ExtractedOutcomeMapping, ExtractedQuestion, ExtractedStudentIdentity, RawScoreCandidate,
    StructuredPageExtraction,
};
use exam_core::domain::{
    BoundingBox, Confidence, LearningOutcomeId, ModelVersion, QuestionId, ReviewIssue,
    ReviewReason, ScoreValue, SourceRegion,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// Crop height handed to the recognizer (the height it was trained on).
const CROP_HEIGHT: u32 = 64;
/// Below this template correlation the sheet location is doubtful and the
/// page goes to review.
const MIN_REGISTRATION_SCORE: f64 = 0.35;

#[derive(Debug, Clone, Copy, Deserialize)]
struct NormalizedBox {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Deserialize)]
struct TemplateQuestion {
    id: String,
    outcome_id: String,
    mapping_cell: NormalizedBox,
    score_cell: NormalizedBox,
}

#[derive(Debug, Deserialize)]
struct TemplateFile {
    template_id: String,
    image: String,
    fields: BTreeMap<String, NormalizedBox>,
    questions: Vec<TemplateQuestion>,
}

#[derive(Debug)]
pub enum SheetError {
    Template(String),
    Image(String),
}

impl fmt::Display for SheetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Template(reason) => write!(f, "invalid sheet template: {reason}"),
            Self::Image(reason) => write!(f, "unreadable image: {reason}"),
        }
    }
}

impl std::error::Error for SheetError {}

pub struct TemplateQuestionCell {
    pub id: QuestionId,
    pub printed_id: String,
    pub outcomes: Vec<LearningOutcomeId>,
    mapping_cell: Rect,
    score_cell: Rect,
}

/// The blank sheet and where each field sits on it (template pixels).
pub struct SheetTemplate {
    pub template_id: String,
    pub questions: Vec<TemplateQuestionCell>,
    name_cell: Rect,
    number_cell: Rect,
    total_cell: Rect,
    size: (usize, usize),
    aligner: Aligner,
    registrar: Registrar,
}

impl SheetTemplate {
    /// Loads `<id>.json` and the blank sheet image next to it. Alignment runs
    /// at half the template resolution, which is close to a phone photo's.
    pub fn load(json_path: impl AsRef<Path>) -> Result<Self, SheetError> {
        let json_path = json_path.as_ref();
        let text = std::fs::read_to_string(json_path)
            .map_err(|e| SheetError::Template(format!("{}: {e}", json_path.display())))?;
        let file: TemplateFile =
            serde_json::from_str(&text).map_err(|e| SheetError::Template(e.to_string()))?;
        let image_path = json_path.with_file_name(&file.image);
        let blank = image::open(&image_path)
            .map_err(|e| SheetError::Template(format!("{}: {e}", image_path.display())))?
            .to_luma8();
        Self::from_blank(file, &blank)
    }

    /// Loads a template from memory, e.g. from app assets on a phone.
    pub fn from_bytes(json: &str, blank_png: &[u8]) -> Result<Self, SheetError> {
        let file: TemplateFile =
            serde_json::from_str(json).map_err(|e| SheetError::Template(e.to_string()))?;
        let blank = image::load_from_memory(blank_png)
            .map_err(|e| SheetError::Template(format!("blank sheet image: {e}")))?
            .to_luma8();
        Self::from_blank(file, &blank)
    }

    fn from_blank(file: TemplateFile, blank: &image::GrayImage) -> Result<Self, SheetError> {
        let half = image::imageops::resize(
            blank,
            blank.width() / 2,
            blank.height() / 2,
            image::imageops::FilterType::Triangle,
        );
        Self::from_parts(file, &half)
    }

    fn from_parts(file: TemplateFile, blank: &image::GrayImage) -> Result<Self, SheetError> {
        let (w, h) = (f64::from(blank.width()), f64::from(blank.height()));
        let rect = |b: &NormalizedBox| Rect {
            x: b.x * w,
            y: b.y * h,
            width: b.width * w,
            height: b.height * h,
        };
        let field = |key: &str| {
            file.fields
                .get(key)
                .map(rect)
                .ok_or_else(|| SheetError::Template(format!("missing field {key}")))
        };
        let questions = file
            .questions
            .iter()
            .map(|q| {
                Ok(TemplateQuestionCell {
                    id: QuestionId::new(&q.id).map_err(|e| SheetError::Template(e.to_string()))?,
                    printed_id: q.id.clone(),
                    outcomes: vec![LearningOutcomeId::new(&q.outcome_id)
                        .map_err(|e| SheetError::Template(e.to_string()))?],
                    mapping_cell: rect(&q.mapping_cell),
                    score_cell: rect(&q.score_cell),
                })
            })
            .collect::<Result<Vec<_>, SheetError>>()?;
        if questions.is_empty() {
            return Err(SheetError::Template("template has no questions".to_owned()));
        }
        let mut template = Self {
            template_id: file.template_id,
            name_cell: field("student.name")?,
            number_cell: field("student.number")?,
            total_cell: field("declared_total")?,
            questions,
            size: (blank.width() as usize, blank.height() as usize),
            aligner: Aligner::from_cells(1, 1, &[]),
            registrar: Registrar::new(blank),
        };
        template.rebuild_aligner();
        Ok(template)
    }

    fn rebuild_aligner(&mut self) {
        let mut cells = vec![self.name_cell, self.number_cell, self.total_cell];
        for q in &self.questions {
            cells.extend([q.mapping_cell, q.score_cell]);
        }
        self.aligner = Aligner::from_cells(self.size.0, self.size.1, &cells);
    }

    /// Re-lays the `Soru-PÇ` table for an exam imported from its Word file:
    /// its questions, PÇ and column widths. The rows and the rest of the
    /// header are the same on every sheet of this form.
    pub fn with_imported_exam(
        mut self,
        exam: &exam_import::ImportedExam,
    ) -> Result<Self, SheetError> {
        let first = self
            .questions
            .first()
            .ok_or_else(|| SheetError::Template("template has no questions".to_owned()))?;
        let (score_row, mapping_row) = (first.score_cell, first.mapping_cell);
        let scale = self.size.0 as f64 / f64::from(exam.page_width_twips);
        let column = |span: exam_import::ColumnSpan, row: Rect| Rect {
            x: f64::from(span.left_twips) * scale,
            width: f64::from(span.width_twips) * scale,
            ..row
        };
        self.questions = exam
            .questions
            .iter()
            .map(|q| TemplateQuestionCell {
                id: q.id.clone(),
                printed_id: q.printed_id.clone(),
                outcomes: q.outcomes.clone(),
                mapping_cell: column(q.score_column, mapping_row),
                score_cell: column(q.score_column, score_row),
            })
            .collect();
        self.total_cell = column(exam.total_column, self.total_cell);
        self.rebuild_aligner();
        Ok(self)
    }
}

/// What a cell of the sheet holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheetField {
    StudentName,
    StudentNumber,
    /// Score cell of the question at this index in the template.
    Score(usize),
    DeclaredTotal,
}

#[derive(Debug, Clone)]
pub struct FieldReading {
    pub field: SheetField,
    /// Benchmark field key, e.g. `question:1.earned_score`.
    pub field_key: String,
    /// Cell corners in photo pixels (TL, TR, BR, BL).
    pub quad: [Point; 4],
    /// `None` for fields that are located but not recognised (the name).
    pub reading: Option<CellReading>,
}

#[derive(Debug, Clone)]
pub struct SheetReading {
    pub width: u32,
    pub height: u32,
    /// False when the printed sheet could not be matched convincingly.
    pub page_found: bool,
    /// Correlation of the photo with the blank template at `page_quad`.
    pub registration_score: f64,
    pub page_quad: [Point; 4],
    pub fields: Vec<FieldReading>,
}

pub struct TemplateSheetAnalyzer {
    template: SheetTemplate,
    recognizer: CellRecognizer,
    maximum_scores: BTreeMap<QuestionId, ScoreValue>,
    model_version: ModelVersion,
}

impl TemplateSheetAnalyzer {
    /// `maximum_scores` comes from the exam definition (they are printed in
    /// the question texts, not in the table).
    pub fn new(
        template: SheetTemplate,
        recognizer: CellRecognizer,
        maximum_scores: BTreeMap<QuestionId, ScoreValue>,
        model_version: ModelVersion,
    ) -> Result<Self, SheetError> {
        if let Some(missing) = template
            .questions
            .iter()
            .find(|q| !maximum_scores.contains_key(&q.id))
        {
            return Err(SheetError::Template(format!(
                "no maximum score for question {}",
                missing.printed_id
            )));
        }
        Ok(Self {
            template,
            recognizer,
            maximum_scores,
            model_version,
        })
    }

    /// Reader for an exam imported from its Word file: the table layout,
    /// questions, PÇ and maximum scores all come from the import.
    pub fn for_imported_exam(
        template: SheetTemplate,
        recognizer: CellRecognizer,
        exam: &exam_import::ImportedExam,
        model_version: ModelVersion,
    ) -> Result<Self, SheetError> {
        let maximum_scores = exam
            .questions
            .iter()
            .map(|q| {
                q.maximum_score.map(|m| (q.id.clone(), m)).ok_or_else(|| {
                    SheetError::Template(format!("no maximum score for question {}", q.printed_id))
                })
            })
            .collect::<Result<_, _>>()?;
        Self::new(
            template.with_imported_exam(exam)?,
            recognizer,
            maximum_scores,
            model_version,
        )
    }

    pub fn template(&self) -> &SheetTemplate {
        &self.template
    }

    pub fn read_sheet(&self, photo: &image::RgbImage) -> Result<SheetReading, AnalyzerError> {
        let (width, height) = photo.dimensions();
        let (w, h) = (f64::from(width), f64::from(height));
        let frame = |inset: f64| {
            Rect {
                x: w * inset,
                y: h * inset,
                width: w * (1.0 - 2.0 * inset),
                height: h * (1.0 - 2.0 * inset),
            }
            .corners()
        };
        let mut candidates: Vec<[Point; 4]> = detect_page(photo).into_iter().collect();
        candidates.extend([frame(0.0), frame(0.06)]);
        let (page_quad, registration_score) = self.template.registrar.register(photo, &candidates);
        let (tw, th) = self.template.aligner.template_size();
        let template_corners = Rect {
            x: 0.0,
            y: 0.0,
            width: tw as f64,
            height: th as f64,
        }
        .corners();
        let to_photo = Homography::from_points(&template_corners, &page_quad)
            .ok_or_else(|| AnalyzerError::Failed("degenerate page outline".to_owned()))?;

        let t = &self.template;
        let student = t
            .aligner
            .align(photo, &to_photo, &[t.name_cell, t.number_cell]);
        let mut table_cells: Vec<Rect> = t.questions.iter().map(|q| q.score_cell).collect();
        table_cells.push(t.total_cell);
        table_cells.extend(t.questions.iter().map(|q| q.mapping_cell));
        let table = t.aligner.align(photo, &to_photo, &table_cells);

        let mut cells = vec![
            (
                SheetField::StudentName,
                "student.name".to_owned(),
                student[0],
                None,
            ),
            (
                SheetField::StudentNumber,
                "student.number".to_owned(),
                student[1],
                Some(CellKind::StudentNumber),
            ),
        ];
        for (index, question) in t.questions.iter().enumerate() {
            cells.push((
                SheetField::Score(index),
                format!("question:{}.earned_score", question.id),
                table[index],
                Some(CellKind::Score),
            ));
        }
        cells.push((
            SheetField::DeclaredTotal,
            "declared_total".to_owned(),
            table[t.questions.len()],
            Some(CellKind::DeclaredTotal),
        ));

        let fields = cells
            .into_iter()
            .map(|(field, field_key, rect, kind)| {
                let quad = rect.corners().map(|p| to_photo.apply(p));
                let reading = kind
                    .map(|kind| {
                        let crop = crop(photo, &quad);
                        let image = CellImage::new(
                            crop.width() as usize,
                            crop.height() as usize,
                            crop.as_raw(),
                        )
                        .map_err(|e| AnalyzerError::Failed(e.to_string()))?;
                        self.recognizer
                            .read(&image, kind)
                            .map_err(|e| AnalyzerError::Failed(e.to_string()))
                    })
                    .transpose()?;
                Ok(FieldReading {
                    field,
                    field_key,
                    quad,
                    reading,
                })
            })
            .collect::<Result<Vec<_>, AnalyzerError>>()?;
        Ok(SheetReading {
            width,
            height,
            page_found: registration_score >= MIN_REGISTRATION_SCORE,
            registration_score,
            page_quad,
            fields,
        })
    }

    /// The port-level extraction for a sheet already read with `read_sheet`.
    pub fn extraction(&self, page_index: u32, sheet: &SheetReading) -> StructuredPageExtraction {
        let certain = Some(Confidence::new(1.0).expect("valid confidence"));
        let empty = || ExtractedField::new(None, None, None, None);
        let region = |quad: &[Point; 4]| source_region(page_index, quad, sheet.width, sheet.height);
        let mut review_issues = Vec::new();
        if !sheet.page_found {
            review_issues.push(ReviewIssue::new(
                ReviewReason::LowConfidence,
                "page outline",
                Some(format!(
                    "printed sheet matched weakly (correlation {:.2})",
                    sheet.registration_score
                )),
                None,
            ));
        }
        let mut number = empty();
        let mut questions: Vec<ExtractedQuestion> = self
            .template
            .questions
            .iter()
            .map(|q| ExtractedQuestion {
                id: ExtractedField::new(
                    Some(q.id.clone()),
                    Some(q.printed_id.clone()),
                    certain,
                    None,
                ),
                maximum_score: ExtractedField::new(
                    Some(self.maximum_scores[&q.id]),
                    Some(self.maximum_scores[&q.id].to_string()),
                    certain,
                    None,
                ),
                score: None,
            })
            .collect();
        let mut declared_total = None;

        for field in &sheet.fields {
            let Some(reading) = &field.reading else {
                continue;
            };
            let source = region(&field.quad);
            let confidence = Confidence::new(reading.confidence.clamp(0.0, 1.0)).ok();
            let name = match field.field {
                SheetField::StudentNumber => "student number".to_owned(),
                SheetField::Score(index) => format!(
                    "question {} score",
                    self.template.questions[index].printed_id
                ),
                SheetField::DeclaredTotal => "declared total".to_owned(),
                SheetField::StudentName => continue,
            };
            if reading.is_blank() {
                review_issues.push(ReviewIssue::new(
                    ReviewReason::BlankField,
                    name,
                    None,
                    source,
                ));
                continue;
            }
            if reading.has_correction() {
                review_issues.push(ReviewIssue::new(
                    ReviewReason::AmbiguousCorrectedScore,
                    name,
                    Some(reading.text.clone()),
                    source.clone(),
                ));
            }
            let value = reading.value_text();
            match field.field {
                SheetField::StudentNumber => {
                    number = ExtractedField::new(
                        Some(value),
                        Some(reading.text.clone()),
                        confidence,
                        source,
                    );
                }
                SheetField::Score(index) => {
                    questions[index].score = Some(RawScoreCandidate {
                        earned: ExtractedField::new(
                            ScoreValue::from_decimal(&value).ok(),
                            Some(reading.text.clone()),
                            confidence,
                            source,
                        ),
                        maximum: None,
                    });
                }
                SheetField::DeclaredTotal => {
                    declared_total = Some(ExtractedField::new(
                        ScoreValue::from_decimal(&value).ok(),
                        Some(reading.text.clone()),
                        confidence,
                        source,
                    ));
                }
                SheetField::StudentName => {}
            }
        }

        let mut outcomes: Vec<LearningOutcomeId> = Vec::new();
        for outcome in self.template.questions.iter().flat_map(|q| &q.outcomes) {
            if !outcomes.contains(outcome) {
                outcomes.push(outcome.clone());
            }
        }
        StructuredPageExtraction {
            page_index,
            expected_page_index: Some(page_index),
            source_page_number: Some(page_index + 1),
            model_version: Some(self.model_version.clone()),
            student: Some(ExtractedStudentIdentity {
                name: empty(),
                student_number: number,
            }),
            questions,
            learning_outcomes: outcomes
                .into_iter()
                .map(|id| ExtractedLearningOutcome {
                    id: ExtractedField::new(
                        Some(id.clone()),
                        Some(id.as_str().to_owned()),
                        certain,
                        None,
                    ),
                    label: empty(),
                })
                .collect(),
            // The form prints `S1-PÇ2` without weights; sessions use
            // `OutcomeAllocationRule::SingleOutcomeImpliesFullWeight`.
            outcome_mappings: self
                .template
                .questions
                .iter()
                .flat_map(|q| q.outcomes.iter().map(move |outcome| (q, outcome)))
                .map(|(q, outcome)| ExtractedOutcomeMapping {
                    question_id: ExtractedField::new(
                        Some(q.id.clone()),
                        Some(q.printed_id.clone()),
                        certain,
                        None,
                    ),
                    outcome_id: ExtractedField::new(
                        Some(outcome.clone()),
                        Some(outcome.as_str().to_owned()),
                        certain,
                        None,
                    ),
                    weight_basis_points: None,
                })
                .collect(),
            declared_total,
            review_issues,
        }
    }
}

impl ExamDocumentAnalyzer for TemplateSheetAnalyzer {
    fn analyze(&self, document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        if document.image_bytes.is_empty() {
            return Err(AnalyzerError::EmptyDocument);
        }
        let photo = image::load_from_memory(&document.image_bytes)
            .map_err(|e| AnalyzerError::Failed(format!("unreadable image: {e}")))?
            .to_rgb8();
        let sheet = self.read_sheet(&photo)?;
        Ok(self.extraction(document.page_index, &sheet))
    }
}

fn crop(photo: &image::RgbImage, quad: &[Point; 4]) -> image::RgbImage {
    let top = (quad[1].x - quad[0].x).hypot(quad[1].y - quad[0].y);
    let side = (quad[3].x - quad[0].x).hypot(quad[3].y - quad[0].y);
    let width = ((f64::from(CROP_HEIGHT) * top / side.max(1.0)).round() as u32).clamp(8, 1024);
    warp_quad(photo, quad, width, CROP_HEIGHT)
}

/// Axis-aligned box of a photo quad, normalized to the photo size.
pub fn source_region(
    page_index: u32,
    quad: &[Point; 4],
    width: u32,
    height: u32,
) -> Option<SourceRegion> {
    let (w, h) = (f64::from(width), f64::from(height));
    let x0 = quad
        .iter()
        .map(|p| p.x)
        .fold(f64::MAX, f64::min)
        .clamp(0.0, w)
        / w;
    let y0 = quad
        .iter()
        .map(|p| p.y)
        .fold(f64::MAX, f64::min)
        .clamp(0.0, h)
        / h;
    let x1 = quad
        .iter()
        .map(|p| p.x)
        .fold(f64::MIN, f64::max)
        .clamp(0.0, w)
        / w;
    let y1 = quad
        .iter()
        .map(|p| p.y)
        .fold(f64::MIN, f64::max)
        .clamp(0.0, h)
        / h;
    BoundingBox::new(x0 as f32, y0 as f32, (x1 - x0) as f32, (y1 - y0) as f32)
        .ok()
        .map(|bounds| SourceRegion::new(page_index, bounds))
}
