use exam_core::domain::{
    Confidence, LearningOutcomeId, ModelVersion, QuestionId, ReviewIssue, ScoreValue, SourceRegion,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentPage {
    pub page_index: u32,
    pub image_bytes: Vec<u8>,
}

impl DocumentPage {
    pub fn new(page_index: u32, image_bytes: Vec<u8>) -> Self {
        Self {
            page_index,
            image_bytes,
        }
    }
}

/// A model-independent candidate. Invalid or partial model output can be kept
/// as raw text without pretending it is already a valid domain value.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedField<T> {
    pub candidate: Option<T>,
    pub raw_text: Option<String>,
    pub confidence: Option<Confidence>,
    pub source_region: Option<SourceRegion>,
}

impl<T> ExtractedField<T> {
    pub fn new(
        candidate: Option<T>,
        raw_text: Option<String>,
        confidence: Option<Confidence>,
        source_region: Option<SourceRegion>,
    ) -> Self {
        Self {
            candidate,
            raw_text,
            confidence,
            source_region,
        }
    }

    pub fn raw_candidate(&self) -> Option<String>
    where
        T: fmt::Display,
    {
        self.raw_text
            .clone()
            .or_else(|| self.candidate.as_ref().map(ToString::to_string))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawScoreCandidate {
    pub earned: ExtractedField<ScoreValue>,
    pub maximum: Option<ExtractedField<ScoreValue>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedStudentIdentity {
    pub name: ExtractedField<String>,
    pub student_number: ExtractedField<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedQuestion {
    pub id: ExtractedField<QuestionId>,
    pub maximum_score: ExtractedField<ScoreValue>,
    pub score: Option<RawScoreCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedLearningOutcome {
    pub id: ExtractedField<LearningOutcomeId>,
    pub label: ExtractedField<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedOutcomeMapping {
    pub question_id: ExtractedField<QuestionId>,
    pub outcome_id: ExtractedField<LearningOutcomeId>,
    /// Hundredths of one percent (10000 == 100%). None means unresolved; it
    /// must never be interpreted as an equal split.
    pub weight_basis_points: Option<ExtractedField<u16>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredPageExtraction {
    pub page_index: u32,
    pub expected_page_index: Option<u32>,
    pub source_page_number: Option<u32>,
    pub model_version: Option<ModelVersion>,
    pub student: Option<ExtractedStudentIdentity>,
    pub questions: Vec<ExtractedQuestion>,
    pub learning_outcomes: Vec<ExtractedLearningOutcome>,
    pub outcome_mappings: Vec<ExtractedOutcomeMapping>,
    /// The handwritten total grade (e.g. the `Not:` cell), if the form has one.
    pub declared_total: Option<ExtractedField<ScoreValue>>,
    pub review_issues: Vec<ReviewIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyzerError {
    EmptyDocument,
    NotConfigured,
    Failed(String),
}

impl fmt::Display for AnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDocument => write!(f, "document page has no bytes"),
            Self::NotConfigured => write!(f, "document analyzer is not configured"),
            Self::Failed(reason) => write!(f, "document analysis failed: {reason}"),
        }
    }
}

impl std::error::Error for AnalyzerError {}

pub trait ExamDocumentAnalyzer {
    fn analyze(&self, document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError>;
}
