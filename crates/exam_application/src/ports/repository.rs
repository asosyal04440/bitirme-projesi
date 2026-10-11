use exam_core::domain::{ExamSession, PageFailure};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryError(pub String);

impl fmt::Display for RepositoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RepositoryError {}

pub trait SessionRepository {
    fn load(&self, session_id: &str) -> Result<ExamSession, RepositoryError>;
    /// Persists the updated session, including newly resolved structure and a
    /// successful page checkpoint, as one repository operation.
    fn save_session(&mut self, session: ExamSession) -> Result<(), RepositoryError>;
    fn record_page_failure(
        &mut self,
        session_id: &str,
        failure: PageFailure,
    ) -> Result<(), RepositoryError>;
}
