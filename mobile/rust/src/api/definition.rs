//! An exam definition the app can store: what was recognised on the sheet
//! (or imported from Word), plus what the instructor confirmed or entered.

use crate::api::exam::{info, ExamInfo};
use exam_core::domain::{LearningOutcomeId, QuestionId, ScoreValue};
use exam_import::{ColumnSpan, ExamMetadata, ImportWarning, ImportedExam, ImportedQuestion};

#[derive(Clone)]
pub struct DefinedQuestion {
    /// As printed, e.g. `S1`.
    pub printed_id: String,
    /// e.g. `PÇ2`; empty when it could not be read and was not entered yet.
    pub outcomes: Vec<String>,
    /// Maximum score, e.g. `25`; `None` until the instructor enters it.
    pub maximum: Option<String>,
    /// The question's column of the `Soru-PÇ` table, in units of
    /// `page_width` (the full page width).
    pub left: u32,
    pub width: u32,
}

#[derive(Clone)]
pub struct ExamDefinition {
    pub course: Option<String>,
    pub course_code: Option<String>,
    pub course_name: Option<String>,
    pub academic_year: Option<String>,
    pub term: Option<String>,
    pub exam_kind: Option<String>,
    pub group: Option<String>,
    pub questions: Vec<DefinedQuestion>,
    pub page_width: u32,
    /// The `Not:` cell, in the same units.
    pub total_left: u32,
    pub total_width: u32,
}

impl ExamDefinition {
    pub(crate) fn from_imported(exam: &ImportedExam) -> Self {
        let m = &exam.metadata;
        Self {
            course: m.course.clone(),
            course_code: m.course_code.clone(),
            course_name: m.course_name.clone(),
            academic_year: m.academic_year.clone(),
            term: m.term.clone(),
            exam_kind: m.exam_kind.clone(),
            group: m.group.clone(),
            questions: exam
                .questions
                .iter()
                .map(|q| DefinedQuestion {
                    printed_id: q.printed_id.clone(),
                    outcomes: q.outcomes.iter().map(|o| o.as_str().to_owned()).collect(),
                    maximum: q.maximum_score.map(|v| v.to_string()),
                    left: q.score_column.left_twips,
                    width: q.score_column.width_twips,
                })
                .collect(),
            page_width: exam.page_width_twips,
            total_left: exam.total_column.left_twips,
            total_width: exam.total_column.width_twips,
        }
    }

    /// Back to the import shape the reader is built from. Questions without
    /// a maximum are reported as warnings.
    pub(crate) fn to_imported(&self) -> anyhow::Result<ImportedExam> {
        let mut questions = Vec::with_capacity(self.questions.len());
        for q in &self.questions {
            let id = QuestionId::new(&q.printed_id).map_err(|e| anyhow::anyhow!("{e}"))?;
            let outcomes = q
                .outcomes
                .iter()
                .map(|o| LearningOutcomeId::new(o.trim()).map_err(|e| anyhow::anyhow!("{o}: {e}")))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let maximum = q
                .maximum
                .as_deref()
                .map(|v| v.trim().replace(',', "."))
                .filter(|v| !v.is_empty())
                .map(|v| {
                    ScoreValue::from_decimal(&v)
                        .map_err(|e| anyhow::anyhow!("{}: {e}", q.printed_id))
                })
                .transpose()?;
            questions.push(ImportedQuestion {
                printed_id: q.printed_id.clone(),
                id,
                outcomes,
                maximum_score: maximum,
                maximum_source: None,
                score_column: ColumnSpan {
                    left_twips: q.left,
                    width_twips: q.width,
                },
            });
        }
        let warnings = questions
            .iter()
            .filter(|q| q.maximum_score.is_none())
            .map(|q| ImportWarning::MissingMaximumScore(q.id.clone()))
            .collect();
        Ok(ImportedExam {
            metadata: ExamMetadata {
                academic_year: self.academic_year.clone(),
                term: self.term.clone(),
                exam_kind: self.exam_kind.clone(),
                course: self.course.clone(),
                course_code: self.course_code.clone(),
                course_name: self.course_name.clone(),
                group: self.group.clone(),
                ..ExamMetadata::default()
            },
            questions,
            page_width_twips: self.page_width,
            total_column: ColumnSpan {
                left_twips: self.total_left,
                width_twips: self.total_width,
            },
            warnings,
        })
    }
}

/// What the app shows for a definition (same as for a Word import).
#[flutter_rust_bridge::frb(sync)]
pub fn definition_info(definition: &ExamDefinition) -> anyhow::Result<ExamInfo> {
    Ok(info(&definition.to_imported()?))
}
