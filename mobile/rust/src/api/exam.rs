//! What the app shows when the instructor adds an exam from its Word file.

use exam_import::{ImportWarning, ImportedExam};

pub struct QuestionInfo {
    /// As printed, e.g. `S1`.
    pub id: String,
    pub outcomes: Vec<String>,
    /// `None` when the Word file does not state it; the instructor enters it.
    pub maximum_score: Option<String>,
    /// The text the maximum was read from, e.g. `(3*10=30 puan)`.
    pub maximum_source: Option<String>,
}

pub struct ExamInfo {
    pub course: Option<String>,
    pub course_code: Option<String>,
    pub course_name: Option<String>,
    pub academic_year: Option<String>,
    pub term: Option<String>,
    pub exam_kind: Option<String>,
    pub group: Option<String>,
    pub date: Option<String>,
    pub instructor: Option<String>,
    pub questions: Vec<QuestionInfo>,
    /// Human-readable (Turkish) notes on what is missing or uncertain.
    pub warnings: Vec<String>,
    /// True when every question has a maximum score.
    pub ready: bool,
}

pub(crate) fn warning_text(warning: &ImportWarning) -> String {
    match warning {
        ImportWarning::MissingMaximumScore(id) => {
            format!("S{id} için tam puan Word dosyasında bulunamadı; lütfen girin.")
        }
        ImportWarning::SummedSubQuestionPoints(id) => {
            format!("S{id} tam puanı alt şıkların puanları toplanarak bulundu; kontrol edin.")
        }
        ImportWarning::UnweightedOutcomes(id) => {
            format!("S{id} birden fazla PÇ ile ilişkili; puan dağılımını belirtin.")
        }
        ImportWarning::QuestionNotInTable(id) => {
            format!("S{id} metinde var ama Soru-PÇ tablosunda yok.")
        }
    }
}

pub(crate) fn info(exam: &ImportedExam) -> ExamInfo {
    let m = &exam.metadata;
    ExamInfo {
        course: m.course.clone(),
        course_code: m.course_code.clone(),
        course_name: m.course_name.clone(),
        academic_year: m.academic_year.clone(),
        term: m.term.clone(),
        exam_kind: m.exam_kind.clone(),
        group: m.group.clone(),
        date: m.date.clone(),
        instructor: m.instructor.clone(),
        questions: exam
            .questions
            .iter()
            .map(|q| QuestionInfo {
                id: q.printed_id.clone(),
                outcomes: q.outcomes.iter().map(|o| o.as_str().to_owned()).collect(),
                maximum_score: q.maximum_score.map(|v| v.to_string()),
                maximum_source: q.maximum_source.clone(),
            })
            .collect(),
        warnings: exam.warnings.iter().map(warning_text).collect(),
        ready: exam.questions.iter().all(|q| q.maximum_score.is_some()),
    }
}

/// Reads an exam definition from a `.docx` file.
pub fn inspect_exam(docx: Vec<u8>) -> anyhow::Result<ExamInfo> {
    Ok(info(&ImportedExam::from_docx(&docx)?))
}
