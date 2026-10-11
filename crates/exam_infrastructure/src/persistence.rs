use exam_application::ports::repository::{RepositoryError, SessionRepository};
use exam_core::domain::{ExamSession, PageFailure};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct InMemorySessionRepository {
    sessions: BTreeMap<String, ExamSession>,
}

impl InMemorySessionRepository {
    pub fn insert(&mut self, session: ExamSession) {
        self.sessions.insert(session.id.clone(), session);
    }

    pub fn get(&self, session_id: &str) -> Option<&ExamSession> {
        self.sessions.get(session_id)
    }
}

impl SessionRepository for InMemorySessionRepository {
    fn load(&self, session_id: &str) -> Result<ExamSession, RepositoryError> {
        self.sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| RepositoryError(format!("session not found: {session_id}")))
    }

    fn save_session(&mut self, session: ExamSession) -> Result<(), RepositoryError> {
        if !self.sessions.contains_key(&session.id) {
            return Err(RepositoryError(format!(
                "session not found: {}",
                session.id
            )));
        }
        self.sessions.insert(session.id.clone(), session);
        Ok(())
    }

    fn record_page_failure(
        &mut self,
        session_id: &str,
        failure: PageFailure,
    ) -> Result<(), RepositoryError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RepositoryError(format!("session not found: {session_id}")))?;
        session.record_failure(failure);
        Ok(())
    }
}
