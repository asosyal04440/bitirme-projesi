use crate::instrumentation::{Instrumentation, Stage};
use crate::ports::analyzer::{
    AnalyzerError, DocumentPage, ExamDocumentAnalyzer, ExtractedField, StructuredPageExtraction,
};
use crate::ports::repository::{RepositoryError, SessionRepository};
use crate::recognition_policy::{RecognitionFieldKind, RecognitionPolicy};
use crate::roster::{RosterMatch, StudentRoster};
use exam_core::domain::{
    DeclaredTotal, DomainError, ExamPage, PageFailure, PageMetadata,
    PendingLearningOutcomeDefinition, PendingOutcomeMapping, PendingQuestionDefinition, QuestionId,
    QuestionResult, ReviewIssue, ReviewReason, Score, ScoreValue, SourceRegion, StudentNumber,
};
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ProcessPageError {
    Analyzer(AnalyzerError),
    Repository(RepositoryError),
    Domain(DomainError),
    FailureRecording {
        primary: Box<ProcessPageError>,
        recording: RepositoryError,
    },
}

impl fmt::Display for ProcessPageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analyzer(error) => write!(f, "{error}"),
            Self::Repository(error) => write!(f, "repository error: {error}"),
            Self::Domain(error) => write!(f, "domain error: {error}"),
            Self::FailureRecording { primary, recording } => write!(
                f,
                "{primary}; additionally failed to record page failure: {recording}"
            ),
        }
    }
}

impl std::error::Error for ProcessPageError {}

pub struct ProcessExamPage<'a, A, R, I> {
    analyzer: &'a A,
    repository: &'a mut R,
    instrumentation: &'a I,
    recognition_policy: RecognitionPolicy,
    roster: Option<&'a StudentRoster>,
}

impl<'a, A, R, I> ProcessExamPage<'a, A, R, I>
where
    A: ExamDocumentAnalyzer,
    R: SessionRepository,
    I: Instrumentation,
{
    pub fn new(analyzer: &'a A, repository: &'a mut R, instrumentation: &'a I) -> Self {
        Self::with_recognition_policy(
            analyzer,
            repository,
            instrumentation,
            RecognitionPolicy::default(),
        )
    }

    pub fn with_recognition_policy(
        analyzer: &'a A,
        repository: &'a mut R,
        instrumentation: &'a I,
        recognition_policy: RecognitionPolicy,
    ) -> Self {
        Self {
            analyzer,
            repository,
            instrumentation,
            recognition_policy,
            roster: None,
        }
    }

    /// Identifies students by matching the recognised number against the
    /// course roster; the name then comes from the roster.
    pub fn with_roster(mut self, roster: &'a StudentRoster) -> Self {
        self.roster = Some(roster);
        self
    }

    pub fn execute(
        &mut self,
        session_id: &str,
        document: DocumentPage,
    ) -> Result<ExamPage, ProcessPageError> {
        let _total = self.instrumentation.begin(Stage::Total);
        let page_index = document.page_index;
        let mut session = self
            .repository
            .load(session_id)
            .map_err(ProcessPageError::Repository)?;

        let extraction = {
            let _span = self.instrumentation.begin(Stage::AiInference);
            match self.analyzer.analyze(&document) {
                Ok(extraction) => extraction,
                Err(error) => {
                    return Err(self.record_failure(
                        session_id,
                        page_index,
                        ProcessPageError::Analyzer(error),
                    ));
                }
            }
        };

        let extraction = {
            let _span = self.instrumentation.begin(Stage::StructuredParsing);
            match self.parse_extraction(page_index, extraction) {
                Ok(extraction) => extraction,
                Err(error) => return Err(self.record_failure(session_id, page_index, error)),
            }
        };

        let page = {
            let _span = self.instrumentation.begin(Stage::DomainValidation);
            match self.to_domain_page(&mut session, extraction) {
                Ok(page) => page,
                Err(error) => return Err(self.record_failure(session_id, page_index, error)),
            }
        };

        if let Err(error) = session.checkpoint_page(page) {
            return Err(self.record_failure(
                session_id,
                page_index,
                ProcessPageError::Domain(error),
            ));
        }
        // Checkpointing can add session-level issues (conflicting scores, total
        // mismatch), so return the stored page rather than the input.
        let page = session.pages()[&page_index].clone();

        let _span = self.instrumentation.begin(Stage::Persistence);
        if let Err(error) = self.repository.save_session(session) {
            return Err(self.record_failure(
                session_id,
                page_index,
                ProcessPageError::Repository(error),
            ));
        }
        Ok(page)
    }

    fn parse_extraction(
        &self,
        document_page_index: u32,
        extraction: StructuredPageExtraction,
    ) -> Result<StructuredPageExtraction, ProcessPageError> {
        if extraction.page_index != document_page_index {
            return Err(ProcessPageError::Domain(DomainError::PageIndexMismatch {
                expected: document_page_index,
                actual: extraction.page_index,
            }));
        }
        Ok(extraction)
    }

    fn to_domain_page(
        &self,
        session: &mut exam_core::domain::ExamSession,
        extraction: StructuredPageExtraction,
    ) -> Result<ExamPage, ProcessPageError> {
        let page_index = extraction.page_index;
        let metadata = PageMetadata::new(
            page_index,
            extraction.expected_page_index,
            extraction.source_page_number,
        );
        let mut review_issues = Vec::new();
        for mut issue in extraction.review_issues.iter().cloned() {
            if issue
                .source_region
                .as_ref()
                .is_some_and(|region| region.page_index != page_index)
            {
                let actual = issue.source_region.take().map(|region| region.page_index);
                issue.reason = ReviewReason::SourceRegionMismatch;
                issue.candidate = Some(format!(
                    "{}; source region belongs to page {}",
                    issue.candidate.unwrap_or_default(),
                    actual.unwrap_or_default()
                ));
            }
            review_issues.push(issue);
        }
        record_extraction_region_issues(&extraction, page_index, &mut review_issues);

        self.resolve_student(
            session,
            extraction.student.clone(),
            page_index,
            &mut review_issues,
        );
        if extraction.student.is_none()
            && session.student().is_none()
            && !review_issues
                .iter()
                .any(|issue| issue.reason == ReviewReason::UnresolvedStudentIdentity)
        {
            review_issues.push(ReviewIssue::new(
                ReviewReason::UnresolvedStudentIdentity,
                "student identity",
                None,
                None,
            ));
        }
        self.merge_exam_discoveries(session, &extraction, page_index, &mut review_issues)?;
        session
            .reconcile_pending_discoveries()
            .map_err(ProcessPageError::Domain)?;

        let pending_student = session.pending_student_identity();
        if pending_student.name_candidates.len() > 1
            || pending_student.number_candidates.len() > 1
            || (session.student().is_some() && !pending_student.is_empty())
        {
            let candidates = pending_student
                .name_candidates
                .iter()
                .cloned()
                .chain(
                    pending_student
                        .number_candidates
                        .iter()
                        .map(|number| number.as_str().to_owned()),
                )
                .collect::<Vec<_>>()
                .join(" / ");
            review_issues.push(ReviewIssue::new(
                ReviewReason::InconsistentPageIdentity,
                "student identity",
                Some(candidates),
                None,
            ));
        }

        let mut question_results = Vec::new();
        for question in &extraction.questions {
            let Some(question_id) = question.id.candidate.clone() else {
                self.review_if_low_confidence(
                    &question.id,
                    RecognitionFieldKind::QuestionId,
                    "question identifier",
                    page_index,
                    &mut review_issues,
                );
                if question.id.raw_candidate().is_some() {
                    review_issues.push(issue_from_field(
                        ReviewReason::UnknownQuestion,
                        "question identifier",
                        &question.id,
                        valid_region(&question.id, page_index),
                    ));
                }
                continue;
            };
            if !self.field_is_accepted(&question.id, RecognitionFieldKind::QuestionId) {
                self.review_if_low_confidence(
                    &question.id,
                    RecognitionFieldKind::QuestionId,
                    "question identifier",
                    page_index,
                    &mut review_issues,
                );
                continue;
            }
            let Some(definition) = session
                .exam()
                .and_then(|exam| exam.questions.get(&question_id))
            else {
                continue;
            };
            let Some(candidate) = &question.score else {
                continue;
            };
            if let Some(maximum) = candidate.maximum.as_ref().and_then(|field| field.candidate) {
                let maximum_field = candidate.maximum.as_ref().expect("checked above");
                if !self.field_is_accepted(maximum_field, RecognitionFieldKind::QuestionMaximum) {
                    self.review_if_low_confidence(
                        maximum_field,
                        RecognitionFieldKind::QuestionMaximum,
                        "score maximum",
                        page_index,
                        &mut review_issues,
                    );
                } else if maximum != definition.maximum_score {
                    review_issues.push(issue_from_field(
                        ReviewReason::InvalidCandidate,
                        "score maximum",
                        maximum_field,
                        valid_region(maximum_field, page_index),
                    ));
                    continue;
                }
            }
            if !self.field_is_accepted(&candidate.earned, RecognitionFieldKind::Score) {
                self.review_if_low_confidence(
                    &candidate.earned,
                    RecognitionFieldKind::Score,
                    &format!("question {question_id} score"),
                    page_index,
                    &mut review_issues,
                );
                continue;
            }
            self.accept_score_candidate(
                &question_id,
                &candidate.earned,
                definition.maximum_score,
                page_index,
                &mut question_results,
                &mut review_issues,
            );
        }

        let declared_total = extraction
            .declared_total
            .as_ref()
            .and_then(|field| self.accept_declared_total(field, page_index, &mut review_issues));

        let mut page = ExamPage::new(metadata, question_results).with_review_issues(review_issues);
        if let Some(declared_total) = declared_total {
            page = page.with_declared_total(declared_total);
        }
        if let Some(model_version) = extraction.model_version {
            page = page.with_model_version(model_version);
        }
        Ok(page)
    }

    fn resolve_student(
        &self,
        session: &mut exam_core::domain::ExamSession,
        extracted: Option<crate::ports::analyzer::ExtractedStudentIdentity>,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) {
        let Some(extracted) = extracted else {
            return;
        };
        let name_region = valid_region(&extracted.name, page_index);
        let number_region = valid_region(&extracted.student_number, page_index);
        if let Some(name) = extracted.name.candidate.clone() {
            if self.field_is_accepted(&extracted.name, RecognitionFieldKind::StudentName) {
                session.add_pending_student_name(Some(name), name_region);
            } else {
                self.review_if_low_confidence(
                    &extracted.name,
                    RecognitionFieldKind::StudentName,
                    "student name",
                    page_index,
                    issues,
                );
            }
        } else if extracted.name.raw_candidate().is_some() {
            self.review_if_low_confidence(
                &extracted.name,
                RecognitionFieldKind::StudentName,
                "student name",
                page_index,
                issues,
            );
            issues.push(issue_from_field(
                ReviewReason::UnresolvedStudentIdentity,
                "student name",
                &extracted.name,
                name_region,
            ));
        }

        if let Some(number) = extracted.student_number.candidate.clone() {
            if let Some(roster) = self.roster {
                self.resolve_from_roster(
                    session,
                    roster,
                    &extracted.student_number,
                    &number,
                    page_index,
                    issues,
                );
                return;
            }
            if !self.field_is_accepted(
                &extracted.student_number,
                RecognitionFieldKind::StudentNumber,
            ) {
                self.review_if_low_confidence(
                    &extracted.student_number,
                    RecognitionFieldKind::StudentNumber,
                    "student number",
                    page_index,
                    issues,
                );
                return;
            }
            match StudentNumber::new(number) {
                Ok(number) => session.add_pending_student_number(Some(number), number_region),
                Err(_) => issues.push(issue_from_field(
                    ReviewReason::UnresolvedStudentIdentity,
                    "student number",
                    &extracted.student_number,
                    number_region,
                )),
            }
        } else if extracted.student_number.raw_candidate().is_some() {
            self.review_if_low_confidence(
                &extracted.student_number,
                RecognitionFieldKind::StudentNumber,
                "student number",
                page_index,
                issues,
            );
            issues.push(issue_from_field(
                ReviewReason::UnresolvedStudentIdentity,
                "student number",
                &extracted.student_number,
                number_region,
            ));
        }
    }

    fn resolve_from_roster(
        &self,
        session: &mut exam_core::domain::ExamSession,
        roster: &StudentRoster,
        field: &ExtractedField<String>,
        recognised: &str,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) {
        let region = valid_region(field, page_index);
        let accepted = self.field_is_accepted(field, RecognitionFieldKind::StudentNumber);
        let propose = |session: &mut exam_core::domain::ExamSession,
                       entry: &crate::roster::RosterEntry| {
            session.add_pending_student_number(Some(entry.number.clone()), region.clone());
            session.add_pending_student_name(Some(entry.name.clone()), None);
        };
        match roster.resolve(recognised) {
            RosterMatch::Exact(entry) => {
                propose(session, entry);
                if !accepted {
                    self.review_if_low_confidence(
                        field,
                        RecognitionFieldKind::StudentNumber,
                        "student number",
                        page_index,
                        issues,
                    );
                }
            }
            RosterMatch::NearMatch(entry) => {
                propose(session, entry);
                issues.push(ReviewIssue::new(
                    ReviewReason::RosterNearMatch,
                    "student number",
                    Some(format!(
                        "read {recognised}; roster {} {}",
                        entry.number.as_str(),
                        entry.name
                    )),
                    region,
                ));
            }
            RosterMatch::Ambiguous(entries) => issues.push(ReviewIssue::new(
                ReviewReason::UnresolvedStudentIdentity,
                "student number",
                Some(format!(
                    "read {recognised}; equally close roster numbers: {}",
                    entries
                        .iter()
                        .map(|entry| entry.number.as_str())
                        .collect::<Vec<_>>()
                        .join(" / ")
                )),
                region,
            )),
            RosterMatch::NotFound => {
                if accepted {
                    if let Ok(number) = StudentNumber::new(recognised) {
                        session.add_pending_student_number(Some(number), region.clone());
                    }
                }
                issues.push(ReviewIssue::new(
                    ReviewReason::UnresolvedStudentIdentity,
                    "student number",
                    Some(format!("{recognised} is not on the course roster")),
                    region,
                ));
            }
        }
    }

    fn merge_exam_discoveries(
        &self,
        session: &mut exam_core::domain::ExamSession,
        extraction: &StructuredPageExtraction,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) -> Result<(), ProcessPageError> {
        for outcome in &extraction.learning_outcomes {
            let Some(id) = outcome.id.candidate.clone() else {
                self.review_if_low_confidence(
                    &outcome.id,
                    RecognitionFieldKind::LearningOutcomeId,
                    "learning outcome identifier",
                    page_index,
                    issues,
                );
                continue;
            };
            if !self.field_is_accepted(&outcome.id, RecognitionFieldKind::LearningOutcomeId) {
                self.review_if_low_confidence(
                    &outcome.id,
                    RecognitionFieldKind::LearningOutcomeId,
                    "learning outcome identifier",
                    page_index,
                    issues,
                );
                continue;
            }
            let label = if self
                .field_is_accepted(&outcome.label, RecognitionFieldKind::LearningOutcomeLabel)
            {
                outcome.label.candidate.clone()
            } else {
                self.review_if_low_confidence(
                    &outcome.label,
                    RecognitionFieldKind::LearningOutcomeLabel,
                    "learning outcome label",
                    page_index,
                    issues,
                );
                None
            };
            let label_unreadable = label.is_none() && has_evidence(&outcome.label);
            let mut pending = PendingLearningOutcomeDefinition::new(
                id,
                label,
                valid_region(&outcome.label, page_index)
                    .or_else(|| valid_region(&outcome.id, page_index)),
            );
            if label_unreadable {
                pending = pending.with_unreadable_label();
            }
            session.add_pending_learning_outcome(pending);
        }

        for extracted in &extraction.questions {
            let Some(id) = extracted.id.candidate.clone() else {
                self.review_if_low_confidence(
                    &extracted.id,
                    RecognitionFieldKind::QuestionId,
                    "question identifier",
                    page_index,
                    issues,
                );
                continue;
            };
            if !self.field_is_accepted(&extracted.id, RecognitionFieldKind::QuestionId) {
                self.review_if_low_confidence(
                    &extracted.id,
                    RecognitionFieldKind::QuestionId,
                    "question identifier",
                    page_index,
                    issues,
                );
                continue;
            }
            let maximum = if self.field_is_accepted(
                &extracted.maximum_score,
                RecognitionFieldKind::QuestionMaximum,
            ) {
                extracted.maximum_score.candidate
            } else {
                self.review_if_low_confidence(
                    &extracted.maximum_score,
                    RecognitionFieldKind::QuestionMaximum,
                    "question maximum score",
                    page_index,
                    issues,
                );
                None
            };
            if let (Some(existing), Some(maximum)) = (
                session.exam().and_then(|exam| exam.questions.get(&id)),
                maximum,
            ) {
                if existing.maximum_score != maximum {
                    issues.push(issue_from_field(
                        ReviewReason::InvalidCandidate,
                        "question maximum score",
                        &extracted.maximum_score,
                        valid_region(&extracted.maximum_score, page_index),
                    ));
                }
            }
            session.add_pending_question_definition(PendingQuestionDefinition::new(
                id,
                maximum,
                valid_region(&extracted.maximum_score, page_index)
                    .or_else(|| valid_region(&extracted.id, page_index)),
            ));
        }

        for mapping in &extraction.outcome_mappings {
            let question_id = mapping.question_id.candidate.clone().filter(|_| {
                self.field_is_accepted(&mapping.question_id, RecognitionFieldKind::OutcomeMapping)
            });
            if question_id.is_none() {
                self.review_if_low_confidence(
                    &mapping.question_id,
                    RecognitionFieldKind::OutcomeMapping,
                    "mapped question",
                    page_index,
                    issues,
                );
            }
            let outcome_id = mapping.outcome_id.candidate.clone().filter(|_| {
                self.field_is_accepted(&mapping.outcome_id, RecognitionFieldKind::OutcomeMapping)
            });
            if outcome_id.is_none() {
                self.review_if_low_confidence(
                    &mapping.outcome_id,
                    RecognitionFieldKind::OutcomeMapping,
                    "mapped learning outcome",
                    page_index,
                    issues,
                );
            }
            let (Some(question_id), Some(outcome_id)) = (question_id, outcome_id) else {
                continue;
            };
            let weight_field = mapping.weight_basis_points.as_ref();
            let allocation = weight_field.and_then(|field| {
                self.field_is_accepted(field, RecognitionFieldKind::OutcomeAllocation)
                    .then_some(field.candidate)
                    .flatten()
            });
            if let Some(field) = weight_field {
                if !self.field_is_accepted(field, RecognitionFieldKind::OutcomeAllocation) {
                    self.review_if_low_confidence(
                        field,
                        RecognitionFieldKind::OutcomeAllocation,
                        "outcome allocation",
                        page_index,
                        issues,
                    );
                }
                if let Some(weight) = field.candidate {
                    if weight == 0 || u32::from(weight) > 10_000 {
                        issues.push(issue_from_field(
                            ReviewReason::AmbiguousOutcomeMapping,
                            "outcome allocation",
                            field,
                            valid_region(field, page_index),
                        ));
                    }
                }
            }
            let source_region = weight_field
                .and_then(|field| valid_region(field, page_index))
                .or_else(|| valid_region(&mapping.outcome_id, page_index))
                .or_else(|| valid_region(&mapping.question_id, page_index));
            let allocation_unreadable =
                allocation.is_none() && weight_field.is_some_and(has_evidence);
            let mut pending =
                PendingOutcomeMapping::new(question_id, outcome_id, allocation, source_region);
            if allocation_unreadable {
                pending = pending.with_unreadable_allocation();
            }
            session.add_pending_outcome_mapping(pending);
        }
        Ok(())
    }

    fn accept_score_candidate(
        &self,
        question_id: &QuestionId,
        earned: &ExtractedField<ScoreValue>,
        maximum: ScoreValue,
        page_index: u32,
        results: &mut Vec<QuestionResult>,
        issues: &mut Vec<ReviewIssue>,
    ) {
        let region = valid_region(earned, page_index);
        if !self.field_is_accepted(earned, RecognitionFieldKind::Score) {
            self.review_if_low_confidence(
                earned,
                RecognitionFieldKind::Score,
                &format!("question {question_id} score"),
                page_index,
                issues,
            );
            return;
        }
        let Some(value) = earned.candidate else {
            issues.push(issue_from_field(
                ReviewReason::InvalidCandidate,
                format!("question {question_id} score"),
                earned,
                region,
            ));
            return;
        };
        let Some(confidence) = earned.confidence else {
            issues.push(issue_from_field(
                ReviewReason::LowConfidence,
                format!("question {question_id} score confidence"),
                earned,
                region,
            ));
            return;
        };
        let Some(source_region) = region else {
            issues.push(issue_from_field(
                ReviewReason::InvalidCandidate,
                format!("question {question_id} score source region"),
                earned,
                None,
            ));
            return;
        };
        match Score::new(value, maximum) {
            Ok(score) => results.push(QuestionResult {
                question_id: question_id.clone(),
                score,
                confidence,
                source_region,
            }),
            Err(DomainError::ScoreExceedsMaximum { .. }) => issues.push(ReviewIssue::new(
                ReviewReason::ScoreExceedsMaximum,
                format!("question {question_id} score"),
                Some(format!(
                    "{} / {maximum}",
                    earned.raw_candidate().unwrap_or_else(|| value.to_string())
                )),
                Some(source_region),
            )),
            Err(_) => issues.push(issue_from_field(
                ReviewReason::InvalidCandidate,
                format!("question {question_id} score"),
                earned,
                Some(source_region),
            )),
        }
    }

    fn accept_declared_total(
        &self,
        field: &ExtractedField<ScoreValue>,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) -> Option<DeclaredTotal> {
        if !self.field_is_accepted(field, RecognitionFieldKind::DeclaredTotal) {
            self.review_if_low_confidence(
                field,
                RecognitionFieldKind::DeclaredTotal,
                "declared total",
                page_index,
                issues,
            );
            return None;
        }
        let region = valid_region(field, page_index);
        match (field.candidate, field.confidence, region) {
            (Some(value), Some(confidence), Some(source_region)) => Some(DeclaredTotal {
                value,
                confidence,
                source_region,
            }),
            (_, _, region) => {
                issues.push(issue_from_field(
                    ReviewReason::InvalidCandidate,
                    "declared total",
                    field,
                    region,
                ));
                None
            }
        }
    }

    fn field_is_accepted<T>(&self, field: &ExtractedField<T>, kind: RecognitionFieldKind) -> bool {
        self.recognition_policy.accepts(kind, field.confidence)
    }

    fn review_if_low_confidence<T: fmt::Display>(
        &self,
        field: &ExtractedField<T>,
        kind: RecognitionFieldKind,
        field_name: &str,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) {
        if has_evidence(field) && !self.field_is_accepted(field, kind) {
            issues.push(issue_from_field(
                ReviewReason::LowConfidence,
                field_name,
                field,
                valid_region(field, page_index),
            ));
        }
    }

    fn record_failure(
        &mut self,
        session_id: &str,
        page_index: u32,
        primary: ProcessPageError,
    ) -> ProcessPageError {
        let _span = self.instrumentation.begin(Stage::Persistence);
        match self.repository.record_page_failure(
            session_id,
            PageFailure::new(page_index, primary.to_string()),
        ) {
            Ok(()) => primary,
            Err(recording) => ProcessPageError::FailureRecording {
                primary: Box::new(primary),
                recording,
            },
        }
    }
}

fn has_evidence<T>(field: &ExtractedField<T>) -> bool {
    field.candidate.is_some()
        || field.raw_text.is_some()
        || field.confidence.is_some()
        || field.source_region.is_some()
}

fn valid_region<T>(field: &ExtractedField<T>, page_index: u32) -> Option<SourceRegion>
where
    T: fmt::Display,
{
    field
        .source_region
        .clone()
        .filter(|region| region.page_index == page_index)
}

fn record_extraction_region_issues(
    extraction: &StructuredPageExtraction,
    page_index: u32,
    issues: &mut Vec<ReviewIssue>,
) {
    fn record<T: fmt::Display>(
        field: &ExtractedField<T>,
        name: &str,
        page_index: u32,
        issues: &mut Vec<ReviewIssue>,
    ) {
        if let Some(region) = field
            .source_region
            .as_ref()
            .filter(|region| region.page_index != page_index)
        {
            issues.push(ReviewIssue::new(
                ReviewReason::SourceRegionMismatch,
                name,
                Some(format!(
                    "{}; region page {} != document page {}",
                    field.raw_candidate().unwrap_or_default(),
                    region.page_index,
                    page_index
                )),
                None,
            ));
        }
    }

    if let Some(student) = &extraction.student {
        record(&student.name, "student name", page_index, issues);
        record(
            &student.student_number,
            "student number",
            page_index,
            issues,
        );
    }
    for question in &extraction.questions {
        record(&question.id, "question identifier", page_index, issues);
        record(
            &question.maximum_score,
            "question maximum score",
            page_index,
            issues,
        );
        if let Some(score) = &question.score {
            record(&score.earned, "question score", page_index, issues);
            if let Some(maximum) = &score.maximum {
                record(maximum, "score maximum", page_index, issues);
            }
        }
    }
    for outcome in &extraction.learning_outcomes {
        record(
            &outcome.id,
            "learning outcome identifier",
            page_index,
            issues,
        );
        record(&outcome.label, "learning outcome label", page_index, issues);
    }
    if let Some(total) = &extraction.declared_total {
        record(total, "declared total", page_index, issues);
    }
    for mapping in &extraction.outcome_mappings {
        record(&mapping.question_id, "mapped question", page_index, issues);
        record(
            &mapping.outcome_id,
            "mapped learning outcome",
            page_index,
            issues,
        );
        if let Some(weight) = &mapping.weight_basis_points {
            record(weight, "outcome allocation", page_index, issues);
        }
    }
}

fn issue_from_field<T>(
    reason: ReviewReason,
    field_name: impl Into<String>,
    field: &ExtractedField<T>,
    source_region: Option<SourceRegion>,
) -> ReviewIssue
where
    T: fmt::Display,
{
    ReviewIssue::new(reason, field_name, field.raw_candidate(), source_region)
}
