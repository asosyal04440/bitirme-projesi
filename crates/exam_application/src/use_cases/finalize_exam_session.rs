use crate::ports::repository::{RepositoryError, SessionRepository};
use exam_core::domain::{
    DomainError, ExamSession, LearningOutcomeId, LearningOutcomeResult, TotalCheck,
};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum FinalizeSessionError {
    Repository(RepositoryError),
    Domain(DomainError),
}

impl fmt::Display for FinalizeSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Repository(error) => write!(f, "repository error: {error}"),
            Self::Domain(error) => write!(f, "domain error: {error}"),
        }
    }
}

impl std::error::Error for FinalizeSessionError {}

#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedSession {
    pub session_id: String,
    pub successful_page_count: usize,
    pub failed_page_count: usize,
    pub missing_page_indexes: Vec<u32>,
    pub review_issue_count: usize,
    pub requires_review: bool,
    pub total_check: TotalCheck,
    pub learning_outcomes: BTreeMap<LearningOutcomeId, LearningOutcomeResult>,
}

pub struct FinalizeExamSession<'a, R> {
    repository: &'a R,
}

impl<'a, R> FinalizeExamSession<'a, R>
where
    R: SessionRepository,
{
    pub fn new(repository: &'a R) -> Self {
        Self { repository }
    }

    pub fn execute(&self, session_id: &str) -> Result<FinalizedSession, FinalizeSessionError> {
        let session: ExamSession = self
            .repository
            .load(session_id)
            .map_err(FinalizeSessionError::Repository)?;
        let missing_page_indexes: Vec<_> = session.missing_page_indexes().into_iter().collect();
        let learning_outcomes = session
            .aggregate_learning_outcomes()
            .map_err(FinalizeSessionError::Domain)?;
        Ok(FinalizedSession {
            session_id: session.id.clone(),
            successful_page_count: session.pages().len(),
            failed_page_count: session.failures().len(),
            review_issue_count: session.review_issue_count(),
            requires_review: session.needs_review() || !missing_page_indexes.is_empty(),
            missing_page_indexes,
            total_check: session.total_check(),
            learning_outcomes,
        })
    }
}
