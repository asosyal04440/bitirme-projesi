use exam_application::instrumentation::{
    DeveloperInstrumentation, Instrumentation, NoopInstrumentation, Stage,
};
use exam_application::ports::analyzer::{
    AnalyzerError, DocumentPage, ExamDocumentAnalyzer, ExtractedField, ExtractedLearningOutcome,
    ExtractedOutcomeMapping, ExtractedQuestion, ExtractedStudentIdentity, RawScoreCandidate,
    StructuredPageExtraction,
};
use exam_application::ports::repository::{RepositoryError, SessionRepository};
use exam_application::recognition_policy::{RecognitionFieldKind, RecognitionPolicy};
use exam_application::roster::{RosterEntry, StudentRoster};
use exam_application::use_cases::{FinalizeExamSession, ProcessExamPage, ProcessPageError};
use exam_core::domain::*;
use std::collections::BTreeMap;

#[derive(Default)]
struct TestRepository {
    sessions: BTreeMap<String, ExamSession>,
    fail_save: bool,
    fail_record: bool,
}

impl TestRepository {
    fn insert(&mut self, session: ExamSession) {
        self.sessions.insert(session.id.clone(), session);
    }
}

impl SessionRepository for TestRepository {
    fn load(&self, session_id: &str) -> Result<ExamSession, RepositoryError> {
        self.sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| RepositoryError(format!("session not found: {session_id}")))
    }

    fn save_session(&mut self, session: ExamSession) -> Result<(), RepositoryError> {
        if self.fail_save {
            return Err(RepositoryError("save failed".to_owned()));
        }
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
        if self.fail_record {
            return Err(RepositoryError("failure checkpoint failed".to_owned()));
        }
        self.sessions
            .get_mut(session_id)
            .ok_or_else(|| RepositoryError(format!("session not found: {session_id}")))?
            .record_failure(failure);
        Ok(())
    }
}

fn region(page_index: u32) -> SourceRegion {
    SourceRegion::new(page_index, BoundingBox::new(0.1, 0.1, 0.5, 0.5).unwrap())
}

fn field<T>(candidate: Option<T>, text: Option<&str>, page_index: u32) -> ExtractedField<T> {
    ExtractedField::new(
        candidate,
        text.map(str::to_owned),
        Some(Confidence::new(0.95).unwrap()),
        Some(region(page_index)),
    )
}

fn lower_confidence<T>(field: &mut ExtractedField<T>) {
    field.confidence = Some(Confidence::new(0.4).unwrap());
}

struct FixedAnalyzer {
    score: ScoreValue,
}

impl FixedAnalyzer {
    fn extraction(&self, page_index: u32) -> StructuredPageExtraction {
        let outcome_id = LearningOutcomeId::new("LO-1").unwrap();
        StructuredPageExtraction {
            page_index,
            expected_page_index: Some(page_index),
            source_page_number: Some(page_index + 1),
            model_version: Some(ModelVersion::new("fixture-analyzer", "1").unwrap()),
            student: Some(ExtractedStudentIdentity {
                name: field(
                    Some("Ada Lovelace".to_owned()),
                    Some("Ada Lovelace"),
                    page_index,
                ),
                student_number: field(Some("123".to_owned()), Some("123"), page_index),
            }),
            questions: vec![ExtractedQuestion {
                id: field(Some(QuestionId::new("1").unwrap()), Some("1"), page_index),
                maximum_score: field(
                    Some(ScoreValue::from_decimal("15").unwrap()),
                    Some("15"),
                    page_index,
                ),
                score: Some(RawScoreCandidate {
                    earned: field(Some(self.score), Some(&self.score.to_string()), page_index),
                    maximum: Some(field(
                        Some(ScoreValue::from_decimal("15").unwrap()),
                        Some("15"),
                        page_index,
                    )),
                }),
            }],
            learning_outcomes: vec![ExtractedLearningOutcome {
                id: field(Some(outcome_id.clone()), Some("LO-1"), page_index),
                label: field(Some("Outcome 1".to_owned()), Some("Outcome 1"), page_index),
            }],
            outcome_mappings: vec![ExtractedOutcomeMapping {
                question_id: field(Some(QuestionId::new("1").unwrap()), Some("1"), page_index),
                outcome_id: field(Some(outcome_id), Some("LO-1"), page_index),
                weight_basis_points: Some(field(Some(10_000), Some("100%"), page_index)),
            }],
            declared_total: None,
            review_issues: Vec::new(),
        }
    }
}

impl ExamDocumentAnalyzer for FixedAnalyzer {
    fn analyze(&self, document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        Ok(self.extraction(document.page_index))
    }
}

struct AlternateAnalyzer(FixedAnalyzer);

impl ExamDocumentAnalyzer for AlternateAnalyzer {
    fn analyze(&self, document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        self.0.analyze(document)
    }
}

struct FailingAnalyzer;

impl ExamDocumentAnalyzer for FailingAnalyzer {
    fn analyze(&self, _document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        Err(AnalyzerError::Failed("read failed".to_owned()))
    }
}

struct DuplicateQuestionAnalyzer;

impl ExamDocumentAnalyzer for DuplicateQuestionAnalyzer {
    fn analyze(&self, document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        let fixed = FixedAnalyzer {
            score: ScoreValue::from_decimal("7").unwrap(),
        };
        let mut extraction = fixed.extraction(document.page_index);
        extraction.questions.push(extraction.questions[0].clone());
        Ok(extraction)
    }
}

fn repository() -> TestRepository {
    let outcome =
        LearningOutcome::new(LearningOutcomeId::new("LO-1").unwrap(), "Outcome 1").unwrap();
    let question = Question::new(
        QuestionId::new("1").unwrap(),
        ScoreValue::from_decimal("15").unwrap(),
        [OutcomeAllocation::new(outcome.id.clone(), 10_000).unwrap()],
    )
    .unwrap();
    let exam = Exam::new("exam-1", "Exam", [question], [outcome]).unwrap();
    let student = StudentIdentity::new("Ada", StudentNumber::new("123").unwrap()).unwrap();
    let mut session = ExamSession::new("session-1", Some(3)).unwrap();
    session.merge_exam_discoveries(exam).unwrap();
    session.resolve_student(student).unwrap();
    let mut repository = TestRepository::default();
    repository.insert(session);
    repository
}

#[test]
fn noop_instrumentation_never_collects_timing() {
    let instrumentation = NoopInstrumentation;
    let guard = instrumentation.begin(Stage::Total);
    drop(guard);
    assert!(instrumentation
        .snapshot()
        .iter()
        .all(|metric| metric.samples == 0 && metric.elapsed_nanos == 0));
}

#[test]
fn developer_instrumentation_records_samples() {
    let instrumentation = DeveloperInstrumentation::default();
    drop(instrumentation.begin(Stage::AiInference));
    assert!(instrumentation.snapshot()[Stage::AiInference as usize].samples > 0);
}

#[test]
fn analyzer_is_replaceable_through_the_application_port() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    let first = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    ProcessExamPage::new(&first, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1, 2, 3]))
        .unwrap();

    let second = AlternateAnalyzer(FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    });
    ProcessExamPage::new(&second, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(1, vec![4, 5, 6]))
        .unwrap();

    let finalized = FinalizeExamSession::new(&repository)
        .execute("session-1")
        .unwrap();
    assert_eq!(finalized.successful_page_count, 2);
    assert_eq!(
        finalized
            .learning_outcomes
            .values()
            .next()
            .unwrap()
            .earned_score
            .to_string(),
        "7"
    );
    assert_eq!(
        repository.sessions["session-1"].canonical_question_results()
            [&QuestionId::new("1").unwrap()]
            .observations
            .len(),
        2
    );
}

#[test]
fn raw_invalid_score_is_preserved_as_a_review_issue() {
    let mut repository = TestRepository::default();
    repository.insert(ExamSession::new("session-1", Some(3)).unwrap());
    let instrumentation = NoopInstrumentation;
    let analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("18").unwrap(),
    };
    let page = ProcessExamPage::new(&analyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

    assert!(page.question_results.is_empty());
    assert_eq!(page.review_issues.len(), 1);
    assert_eq!(
        page.review_issues[0].reason,
        ReviewReason::ScoreExceedsMaximum
    );
    assert_eq!(page.review_issues[0].candidate.as_deref(), Some("18 / 15"));
    assert_eq!(
        page.review_issues[0]
            .source_region
            .as_ref()
            .unwrap()
            .page_index,
        0
    );
}

#[test]
fn unresolved_student_identity_is_retained_and_complete_identity_resolves_session() {
    let instrumentation = NoopInstrumentation;
    let fixed_analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    let mut unresolved_repository = TestRepository::default();
    unresolved_repository.insert(ExamSession::new("session-1", Some(1)).unwrap());
    let mut extraction = fixed_analyzer.extraction(0);
    extraction.student = None;
    let analyzer = StaticAnalyzer(extraction);
    let page = ProcessExamPage::new(&analyzer, &mut unresolved_repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();
    assert!(page
        .review_issues
        .iter()
        .any(|issue| issue.reason == ReviewReason::UnresolvedStudentIdentity));
    assert!(unresolved_repository.sessions["session-1"]
        .student()
        .is_none());

    let mut resolved_repository = TestRepository::default();
    resolved_repository.insert(ExamSession::new("session-2", Some(1)).unwrap());
    let fixed_analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    let analyzer = StaticAnalyzer(fixed_analyzer.extraction(0));
    let page = ProcessExamPage::new(&analyzer, &mut resolved_repository, &instrumentation)
        .execute("session-2", DocumentPage::new(0, vec![1]))
        .unwrap();
    assert!(!page
        .review_issues
        .iter()
        .any(|issue| issue.reason == ReviewReason::UnresolvedStudentIdentity));
    assert!(resolved_repository.sessions["session-2"]
        .student()
        .is_some());
}

struct StaticAnalyzer(StructuredPageExtraction);

impl ExamDocumentAnalyzer for StaticAnalyzer {
    fn analyze(&self, _document: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        Ok(self.0.clone())
    }
}

#[test]
fn analyzer_failure_is_recorded_without_erasing_successful_pages() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    let successful = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    ProcessExamPage::new(&successful, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

    let error = ProcessExamPage::new(&FailingAnalyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(1, vec![2]))
        .unwrap_err();
    assert_eq!(
        error,
        ProcessPageError::Analyzer(AnalyzerError::Failed("read failed".to_owned()))
    );
    assert_eq!(repository.sessions["session-1"].pages().len(), 1);
    assert_eq!(repository.sessions["session-1"].failures().len(), 1);
}

#[test]
fn domain_validation_failure_is_checkpointed_and_successful_page_is_kept() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    let successful = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    ProcessExamPage::new(&successful, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

    let error = ProcessExamPage::new(
        &DuplicateQuestionAnalyzer,
        &mut repository,
        &instrumentation,
    )
    .execute("session-1", DocumentPage::new(1, vec![2]))
    .unwrap_err();
    assert!(matches!(
        error,
        ProcessPageError::Domain(DomainError::DuplicateQuestion(_))
    ));
    assert_eq!(repository.sessions["session-1"].pages().len(), 1);
    assert!(repository.sessions["session-1"].failures().contains_key(&1));
}

#[test]
fn failed_page_can_be_retried_and_successful_retry_clears_failure() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    ProcessExamPage::new(&FailingAnalyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap_err();
    assert!(repository.sessions["session-1"].failures().contains_key(&0));

    let successful = FixedAnalyzer {
        score: ScoreValue::from_decimal("7.5").unwrap(),
    };
    ProcessExamPage::new(&successful, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();
    assert!(repository.sessions["session-1"].failures().is_empty());
    assert!(repository.sessions["session-1"].pages().contains_key(&0));
}

#[test]
fn checkpointed_page_retains_model_provenance() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    let analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    ProcessExamPage::new(&analyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();
    let version = repository.sessions["session-1"].pages()[&0]
        .model_version
        .as_ref()
        .unwrap();
    assert_eq!(version.name, "fixture-analyzer");
    assert_eq!(version.version, "1");
}

#[test]
fn process_records_distinct_stage_metrics() {
    let mut repository = repository();
    let instrumentation = DeveloperInstrumentation::default();
    let analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    ProcessExamPage::new(&analyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();
    let metrics = instrumentation.snapshot();
    for stage in [
        Stage::AiInference,
        Stage::StructuredParsing,
        Stage::DomainValidation,
        Stage::Persistence,
        Stage::Total,
    ] {
        assert!(
            metrics[stage as usize].samples > 0,
            "{stage:?} was not recorded"
        );
    }
}

#[test]
fn failure_recording_error_preserves_the_primary_failure() {
    let mut repository = repository();
    repository.fail_record = true;
    let instrumentation = NoopInstrumentation;
    let error = ProcessExamPage::new(&FailingAnalyzer, &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap_err();
    assert!(matches!(
        error,
        ProcessPageError::FailureRecording {
            primary,
            recording: RepositoryError(ref message)
        } if matches!(*primary, ProcessPageError::Analyzer(AnalyzerError::Failed(_)))
            && message == "failure checkpoint failed"
    ));
}

#[test]
fn page_discoveries_reconcile_across_pages_and_student_fields_can_be_split() {
    let mut repository = TestRepository::default();
    repository.insert(ExamSession::new("session-1", Some(2)).unwrap());
    let instrumentation = NoopInstrumentation;
    let outcome_id = LearningOutcomeId::new("LO-A").unwrap();
    let question_id = QuestionId::new("Q3-b").unwrap();

    let mut first = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    }
    .extraction(0);
    first.student.as_mut().unwrap().student_number = ExtractedField::new(None, None, None, None);
    first.questions[0].id.candidate = Some(question_id.clone());
    first.questions[0].id.raw_text = Some("Q3-b".to_owned());
    first.questions[0].maximum_score = ExtractedField::new(None, None, None, None);
    first.questions[0].score = None;
    first.learning_outcomes[0].id.candidate = Some(outcome_id.clone());
    first.learning_outcomes[0].id.raw_text = Some("LO-A".to_owned());
    first.outcome_mappings[0].question_id.candidate = Some(question_id.clone());
    first.outcome_mappings[0].question_id.raw_text = Some("Q3-b".to_owned());
    first.outcome_mappings[0].outcome_id.candidate = Some(outcome_id.clone());
    first.outcome_mappings[0].outcome_id.raw_text = Some("LO-A".to_owned());
    ProcessExamPage::new(&StaticAnalyzer(first), &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

    let mut second = FixedAnalyzer {
        score: ScoreValue::from_decimal("12").unwrap(),
    }
    .extraction(1);
    second.student.as_mut().unwrap().name = ExtractedField::new(None, None, None, None);
    second.questions[0].id.candidate = Some(question_id.clone());
    second.questions[0].id.raw_text = Some("Q3-b".to_owned());
    second.questions[0].maximum_score.candidate = Some(ScoreValue::from_decimal("20").unwrap());
    second.questions[0].maximum_score.raw_text = Some("20".to_owned());
    let score = second.questions[0].score.as_mut().unwrap();
    score.maximum.as_mut().unwrap().candidate = Some(ScoreValue::from_decimal("20").unwrap());
    score.maximum.as_mut().unwrap().raw_text = Some("20".to_owned());
    second.learning_outcomes.clear();
    second.outcome_mappings.clear();
    ProcessExamPage::new(&StaticAnalyzer(second), &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(1, vec![2]))
        .unwrap();

    let session = &repository.sessions["session-1"];
    assert_eq!(session.pending_discovery_count(), 0);
    assert_eq!(session.student().unwrap().name, "Ada Lovelace");
    let exam = session.exam().unwrap();
    assert_eq!(exam.questions[&question_id].maximum_score.to_string(), "20");
    assert_eq!(exam.questions[&question_id].outcome_allocations.len(), 1);
    assert_eq!(
        exam.questions[&question_id].outcome_allocations[0].outcome_id,
        outcome_id
    );
    let aggregate = session.aggregate_learning_outcomes().unwrap();
    assert_eq!(aggregate[&outcome_id].earned_score.to_string(), "12");
    assert_eq!(aggregate[&outcome_id].possible_score.to_string(), "20");
}

#[test]
fn question_definition_before_mapping_reconciles_without_reprocessing_first_page() {
    let mut repository = TestRepository::default();
    repository.insert(ExamSession::new("session-1", Some(2)).unwrap());
    let instrumentation = NoopInstrumentation;
    let question_id = QuestionId::new("Q3-b").unwrap();
    let outcome_id = LearningOutcomeId::new("LO-A").unwrap();

    let mut first = FixedAnalyzer {
        score: ScoreValue::from_decimal("12").unwrap(),
    }
    .extraction(0);
    first.questions[0].id.candidate = Some(question_id.clone());
    first.questions[0].id.raw_text = Some("Q3-b".to_owned());
    first.questions[0].maximum_score.candidate = Some(ScoreValue::from_decimal("20").unwrap());
    first.questions[0].maximum_score.raw_text = Some("20".to_owned());
    let score = first.questions[0].score.as_mut().unwrap();
    score.maximum.as_mut().unwrap().candidate = Some(ScoreValue::from_decimal("20").unwrap());
    score.maximum.as_mut().unwrap().raw_text = Some("20".to_owned());
    first.learning_outcomes[0].id.candidate = Some(outcome_id.clone());
    first.learning_outcomes[0].id.raw_text = Some("LO-A".to_owned());
    first.outcome_mappings.clear();
    ProcessExamPage::new(&StaticAnalyzer(first), &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

    assert!(
        repository.sessions["session-1"].exam().unwrap().questions[&question_id]
            .outcome_allocations
            .is_empty()
    );

    let mut second = FixedAnalyzer {
        score: ScoreValue::from_decimal("12").unwrap(),
    }
    .extraction(1);
    second.student = None;
    second.questions.clear();
    second.learning_outcomes.clear();
    second.outcome_mappings[0].question_id.candidate = Some(question_id.clone());
    second.outcome_mappings[0].question_id.raw_text = Some("Q3-b".to_owned());
    second.outcome_mappings[0].outcome_id.candidate = Some(outcome_id.clone());
    second.outcome_mappings[0].outcome_id.raw_text = Some("LO-A".to_owned());
    ProcessExamPage::new(&StaticAnalyzer(second), &mut repository, &instrumentation)
        .execute("session-1", DocumentPage::new(1, vec![2]))
        .unwrap();

    let session = &repository.sessions["session-1"];
    assert_eq!(session.pending_discovery_count(), 0);
    assert_eq!(
        session.exam().unwrap().questions[&question_id]
            .outcome_allocations
            .len(),
        1
    );
    let aggregate = session.aggregate_learning_outcomes().unwrap();
    assert_eq!(aggregate[&outcome_id].earned_score.to_string(), "12");
    assert_eq!(aggregate[&outcome_id].possible_score.to_string(), "20");
}

#[test]
fn matching_partial_student_observations_do_not_create_identity_conflicts() {
    #[derive(Clone, Copy)]
    enum PartialField {
        Name,
        Number,
    }

    for partial in [PartialField::Name, PartialField::Number] {
        let mut repository = repository();
        let instrumentation = NoopInstrumentation;
        let analyzer = FixedAnalyzer {
            score: ScoreValue::from_decimal("7").unwrap(),
        };
        let mut extraction = analyzer.extraction(0);
        match partial {
            PartialField::Name => {
                let student = extraction.student.as_mut().unwrap();
                student.name.candidate = Some("Ada".to_owned());
                student.name.raw_text = Some("Ada".to_owned());
                student.student_number = ExtractedField::new(None, None, None, None);
            }
            PartialField::Number => {
                extraction.student.as_mut().unwrap().name =
                    ExtractedField::new(None, None, None, None);
            }
        }
        let page = ProcessExamPage::new(
            &StaticAnalyzer(extraction),
            &mut repository,
            &instrumentation,
        )
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

        assert!(!page
            .review_issues
            .iter()
            .any(|issue| issue.reason == ReviewReason::InconsistentPageIdentity));
        let session = &repository.sessions["session-1"];
        assert!(session.pending_student_identity().is_empty());
        assert_eq!(session.student().unwrap().name, "Ada");
        assert_eq!(session.student().unwrap().number.as_str(), "123");
    }
}

#[test]
fn mismatched_partial_student_observations_are_flagged_for_review() {
    for (name, number) in [(Some("Grace Hopper"), None), (None, Some("999"))] {
        let mut repository = repository();
        let instrumentation = NoopInstrumentation;
        let analyzer = FixedAnalyzer {
            score: ScoreValue::from_decimal("7").unwrap(),
        };
        let mut extraction = analyzer.extraction(0);
        let student = extraction.student.as_mut().unwrap();
        if let Some(name) = name {
            student.name.candidate = Some(name.to_owned());
            student.name.raw_text = Some(name.to_owned());
            student.student_number = ExtractedField::new(None, None, None, None);
        } else if let Some(number) = number {
            student.name = ExtractedField::new(None, None, None, None);
            student.student_number.candidate = Some(number.to_owned());
            student.student_number.raw_text = Some(number.to_owned());
        }
        let page = ProcessExamPage::new(
            &StaticAnalyzer(extraction),
            &mut repository,
            &instrumentation,
        )
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();

        assert!(page
            .review_issues
            .iter()
            .any(|issue| issue.reason == ReviewReason::InconsistentPageIdentity));
        assert_eq!(
            repository.sessions["session-1"].student().unwrap().name,
            "Ada"
        );
    }
}

#[test]
fn configurable_score_threshold_retains_low_confidence_candidate_for_review() {
    let mut repository = repository();
    let instrumentation = NoopInstrumentation;
    let analyzer = FixedAnalyzer {
        score: ScoreValue::from_decimal("7").unwrap(),
    };
    let mut extraction = analyzer.extraction(0);
    extraction.questions[0]
        .score
        .as_mut()
        .unwrap()
        .earned
        .confidence = Some(Confidence::new(0.95).unwrap());
    let policy = RecognitionPolicy::default()
        .with_minimum_confidence(RecognitionFieldKind::Score, 0.99)
        .unwrap();
    let page = ProcessExamPage::with_recognition_policy(
        &StaticAnalyzer(extraction),
        &mut repository,
        &instrumentation,
        policy,
    )
    .execute("session-1", DocumentPage::new(0, vec![1]))
    .unwrap();

    assert!(page.question_results.is_empty());
    assert!(page.review_issues.iter().any(|issue| {
        issue.reason == ReviewReason::LowConfidence && issue.candidate.as_deref() == Some("7")
    }));
    assert!(repository.sessions["session-1"].needs_review());
    assert!(repository.sessions["session-1"]
        .aggregate_learning_outcomes()
        .unwrap()
        .is_empty());
}

#[test]
fn low_confidence_policy_gates_every_extraction_field_and_preserves_candidates() {
    #[derive(Debug, Clone, Copy)]
    enum FieldCase {
        StudentName,
        StudentNumber,
        QuestionId,
        QuestionMaximum,
        Score,
        LearningOutcomeId,
        LearningOutcomeLabel,
        OutcomeMappingQuestion,
        OutcomeMappingOutcome,
        OutcomeAllocation,
    }

    let cases = [
        (FieldCase::StudentName, "Ada Lovelace"),
        (FieldCase::StudentNumber, "123"),
        (FieldCase::QuestionId, "1"),
        (FieldCase::QuestionMaximum, "15"),
        (FieldCase::Score, "7"),
        (FieldCase::LearningOutcomeId, "LO-1"),
        (FieldCase::LearningOutcomeLabel, "Outcome 1"),
        (FieldCase::OutcomeMappingQuestion, "1"),
        (FieldCase::OutcomeMappingOutcome, "LO-1"),
        (FieldCase::OutcomeAllocation, "100%"),
    ];
    let question_id = QuestionId::new("1").unwrap();
    let outcome_id = LearningOutcomeId::new("LO-1").unwrap();

    for (field_case, expected_candidate) in cases {
        let analyzer = FixedAnalyzer {
            score: ScoreValue::from_decimal("7").unwrap(),
        };
        let mut extraction = analyzer.extraction(0);
        match field_case {
            FieldCase::StudentName => {
                lower_confidence(&mut extraction.student.as_mut().unwrap().name)
            }
            FieldCase::StudentNumber => {
                lower_confidence(&mut extraction.student.as_mut().unwrap().student_number)
            }
            FieldCase::QuestionId => lower_confidence(&mut extraction.questions[0].id),
            FieldCase::QuestionMaximum => {
                lower_confidence(&mut extraction.questions[0].maximum_score)
            }
            FieldCase::Score => {
                lower_confidence(&mut extraction.questions[0].score.as_mut().unwrap().earned)
            }
            FieldCase::LearningOutcomeId => {
                lower_confidence(&mut extraction.learning_outcomes[0].id)
            }
            FieldCase::LearningOutcomeLabel => {
                lower_confidence(&mut extraction.learning_outcomes[0].label)
            }
            FieldCase::OutcomeMappingQuestion => {
                lower_confidence(&mut extraction.outcome_mappings[0].question_id)
            }
            FieldCase::OutcomeMappingOutcome => {
                lower_confidence(&mut extraction.outcome_mappings[0].outcome_id)
            }
            FieldCase::OutcomeAllocation => lower_confidence(
                extraction.outcome_mappings[0]
                    .weight_basis_points
                    .as_mut()
                    .unwrap(),
            ),
        }

        let mut repository = TestRepository::default();
        repository.insert(ExamSession::new("session-1", Some(1)).unwrap());
        let instrumentation = NoopInstrumentation;
        let page = ProcessExamPage::new(
            &StaticAnalyzer(extraction),
            &mut repository,
            &instrumentation,
        )
        .execute("session-1", DocumentPage::new(0, vec![1]))
        .unwrap();
        assert!(
            page.review_issues.iter().any(|issue| {
                issue.reason == ReviewReason::LowConfidence
                    && issue.candidate.as_deref() == Some(expected_candidate)
            }),
            "{field_case:?} candidate was not retained as a low-confidence issue"
        );

        let session = &repository.sessions["session-1"];
        match field_case {
            FieldCase::StudentName => {
                assert!(session.student().is_none());
                assert!(session
                    .pending_student_identity()
                    .name_candidates
                    .is_empty());
            }
            FieldCase::StudentNumber => {
                assert!(session.student().is_none());
                assert!(session
                    .pending_student_identity()
                    .number_candidates
                    .is_empty());
            }
            FieldCase::QuestionId | FieldCase::QuestionMaximum => {
                assert!(!session.exam().unwrap().questions.contains_key(&question_id));
                if matches!(field_case, FieldCase::QuestionMaximum) {
                    assert!(session.pending_questions()[&question_id]
                        .resolved_maximum()
                        .is_none());
                }
            }
            FieldCase::Score => assert!(session.canonical_question_results().is_empty()),
            FieldCase::LearningOutcomeId => {
                assert!(!session
                    .exam()
                    .unwrap()
                    .learning_outcomes
                    .contains_key(&outcome_id));
                assert!(!session
                    .pending_learning_outcomes()
                    .contains_key(&outcome_id));
            }
            FieldCase::LearningOutcomeLabel => {
                assert!(!session
                    .exam()
                    .unwrap()
                    .learning_outcomes
                    .contains_key(&outcome_id));
                assert!(session.pending_learning_outcomes()[&outcome_id]
                    .resolved_label()
                    .is_none());
            }
            FieldCase::OutcomeMappingQuestion | FieldCase::OutcomeMappingOutcome => {
                assert!(session.pending_outcome_mappings().is_empty());
                assert!(session.exam().unwrap().questions[&question_id]
                    .outcome_allocations
                    .is_empty());
            }
            FieldCase::OutcomeAllocation => {
                assert!(session.pending_outcome_mappings()
                    [&(question_id.clone(), outcome_id.clone())]
                    .resolved_allocation()
                    .is_none());
            }
        }
    }
}

fn empty<T>() -> ExtractedField<T> {
    ExtractedField::new(None, None, None, None)
}

/// Page 1 of the İnönü exam sheet: handwritten identity, the printed
/// `Soru-PÇ İlişkisi` table without weights or labels, the `Puan:` row and the
/// handwritten `Not:` total.
fn inonu_form_extraction(total: &str) -> StructuredPageExtraction {
    let question = |raw: &str, maximum: &str, earned: &str| ExtractedQuestion {
        id: field(Some(QuestionId::new(raw).unwrap()), Some(raw), 0),
        maximum_score: field(
            Some(ScoreValue::from_decimal(maximum).unwrap()),
            Some(maximum),
            0,
        ),
        score: Some(RawScoreCandidate {
            earned: field(
                Some(ScoreValue::from_decimal(earned).unwrap()),
                Some(earned),
                0,
            ),
            maximum: None,
        }),
    };
    let outcome = |raw: &str| ExtractedLearningOutcome {
        id: field(Some(LearningOutcomeId::new(raw).unwrap()), Some(raw), 0),
        label: empty(),
    };
    let mapping = |question: &str, outcome: &str| ExtractedOutcomeMapping {
        question_id: field(Some(QuestionId::new(question).unwrap()), Some(question), 0),
        outcome_id: field(
            Some(LearningOutcomeId::new(outcome).unwrap()),
            Some(outcome),
            0,
        ),
        weight_basis_points: None,
    };
    StructuredPageExtraction {
        page_index: 0,
        expected_page_index: Some(0),
        source_page_number: Some(1),
        model_version: None,
        student: Some(ExtractedStudentIdentity {
            name: field(Some("Ada Lovelace".to_owned()), Some("Ada Lovelace"), 0),
            student_number: field(Some("12345".to_owned()), Some("12345"), 0),
        }),
        questions: vec![question("(S1)", "20", "15"), question("(S2)", "30", "21")],
        learning_outcomes: vec![outcome("PÇ2"), outcome("PÇ1")],
        outcome_mappings: vec![mapping("(S1)", "PÇ2"), mapping("S.2.", "PÇ1")],
        declared_total: Some(field(
            Some(ScoreValue::from_decimal(total).unwrap()),
            Some(total),
            0,
        )),
        review_issues: Vec::new(),
    }
}

fn process_inonu_form(extraction: StructuredPageExtraction) -> (TestRepository, ExamPage) {
    let mut repository = TestRepository::default();
    repository.insert(
        ExamSession::new("session-1", Some(1))
            .unwrap()
            .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight),
    );
    let instrumentation = NoopInstrumentation;
    let page = ProcessExamPage::new(
        &StaticAnalyzer(extraction),
        &mut repository,
        &instrumentation,
    )
    .execute("session-1", DocumentPage::new(0, vec![1]))
    .unwrap();
    (repository, page)
}

#[test]
fn inonu_form_page_resolves_outcomes_without_weights_or_labels() {
    let (repository, page) = process_inonu_form(inonu_form_extraction("36"));
    assert!(page.review_issues.is_empty(), "{:?}", page.review_issues);
    assert_eq!(
        page.declared_total.as_ref().unwrap().value,
        ScoreValue::from_decimal("36").unwrap()
    );

    let finalized = FinalizeExamSession::new(&repository)
        .execute("session-1")
        .unwrap();
    assert!(!finalized.requires_review);
    assert_eq!(finalized.total_check, TotalCheck::Consistent);
    let pc2 = &finalized.learning_outcomes[&LearningOutcomeId::new("PÇ2").unwrap()];
    assert_eq!(pc2.earned_score.to_string(), "15");
    assert_eq!(pc2.possible_score.to_string(), "20");
    let pc1 = &finalized.learning_outcomes[&LearningOutcomeId::new("PÇ1").unwrap()];
    assert_eq!(pc1.earned_score.to_string(), "21");
    assert_eq!(pc1.possible_score.to_string(), "30");
}

#[test]
fn inonu_form_total_that_disagrees_with_scores_goes_to_review() {
    let (repository, page) = process_inonu_form(inonu_form_extraction("35"));
    assert!(page
        .review_issues
        .iter()
        .any(|issue| issue.reason == ReviewReason::TotalMismatch));

    let finalized = FinalizeExamSession::new(&repository)
        .execute("session-1")
        .unwrap();
    assert!(finalized.requires_review);
    assert_eq!(
        finalized.total_check,
        TotalCheck::Mismatch {
            declared: ScoreValue::from_decimal("35").unwrap(),
            computed: ScoreValue::from_decimal("36").unwrap(),
        }
    );
}

#[test]
fn low_confidence_total_is_kept_for_review_instead_of_checked() {
    let mut extraction = inonu_form_extraction("36");
    lower_confidence(extraction.declared_total.as_mut().unwrap());
    let (repository, page) = process_inonu_form(extraction);

    assert!(page.declared_total.is_none());
    assert!(page.review_issues.iter().any(|issue| {
        issue.reason == ReviewReason::LowConfidence
            && issue.field == "declared total"
            && issue.candidate.as_deref() == Some("36")
    }));
    let finalized = FinalizeExamSession::new(&repository)
        .execute("session-1")
        .unwrap();
    assert_eq!(finalized.total_check, TotalCheck::NotDeclared);
    assert!(finalized.requires_review);
}

#[test]
fn unreadable_printed_weight_is_not_replaced_by_the_single_outcome_rule() {
    let mut extraction = inonu_form_extraction("36");
    let mut weight = field(Some(10_000), Some("100%"), 0);
    lower_confidence(&mut weight);
    extraction.outcome_mappings[0].weight_basis_points = Some(weight);
    let (repository, _) = process_inonu_form(extraction);

    let session = &repository.sessions["session-1"];
    let question = QuestionId::new("1").unwrap();
    assert!(session.exam().unwrap().questions[&question]
        .outcome_allocations
        .is_empty());
    assert!(session.needs_review());
}

fn roster() -> StudentRoster {
    StudentRoster::new([
        RosterEntry {
            number: StudentNumber::new("04237384350").unwrap(),
            name: "Ayşe Yılmaz".to_owned(),
        },
        RosterEntry {
            number: StudentNumber::new("00724475861").unwrap(),
            name: "Mehmet Kaya".to_owned(),
        },
        RosterEntry {
            number: StudentNumber::new("00724475811").unwrap(),
            name: "Can Demir".to_owned(),
        },
    ])
    .unwrap()
}

/// The sheet reader never reads names: only the number is extracted.
fn number_only_extraction(number: &str, confidence: f32) -> StructuredPageExtraction {
    let mut extraction = inonu_form_extraction("36");
    let student = extraction.student.as_mut().unwrap();
    student.name = empty();
    student.student_number = field(Some(number.to_owned()), Some(number), 0);
    student.student_number.confidence = Some(Confidence::new(confidence).unwrap());
    extraction
}

fn process_with_roster(extraction: StructuredPageExtraction) -> (TestRepository, ExamPage) {
    let roster = roster();
    let mut repository = TestRepository::default();
    repository.insert(
        ExamSession::new("session-1", Some(1))
            .unwrap()
            .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight),
    );
    let instrumentation = NoopInstrumentation;
    let page = ProcessExamPage::new(
        &StaticAnalyzer(extraction),
        &mut repository,
        &instrumentation,
    )
    .with_roster(&roster)
    .execute("session-1", DocumentPage::new(0, vec![1]))
    .unwrap();
    (repository, page)
}

#[test]
fn exact_roster_match_takes_the_name_from_the_roster() {
    let (repository, page) = process_with_roster(number_only_extraction("04237384350", 0.95));
    assert!(page.review_issues.is_empty(), "{:?}", page.review_issues);
    let student = repository.sessions["session-1"].student().unwrap();
    assert_eq!(student.name, "Ayşe Yılmaz");
    assert_eq!(student.number.as_str(), "04237384350");
    let finalized = FinalizeExamSession::new(&repository)
        .execute("session-1")
        .unwrap();
    assert!(!finalized.requires_review);
}

#[test]
fn low_confidence_exact_roster_match_is_proposed_but_reviewed() {
    let (repository, page) = process_with_roster(number_only_extraction("04237384350", 0.6));
    assert!(page
        .review_issues
        .iter()
        .any(|issue| issue.reason == ReviewReason::LowConfidence));
    assert_eq!(
        repository.sessions["session-1"].student().unwrap().name,
        "Ayşe Yılmaz"
    );
    assert!(repository.sessions["session-1"].needs_review());
}

#[test]
fn single_misread_digit_proposes_the_unique_roster_student_for_review() {
    let (repository, page) = process_with_roster(number_only_extraction("04237384950", 0.95));
    let issue = page
        .review_issues
        .iter()
        .find(|issue| issue.reason == ReviewReason::RosterNearMatch)
        .expect("near match issue");
    assert_eq!(
        issue.candidate.as_deref(),
        Some("read 04237384950; roster 04237384350 Ayşe Yılmaz")
    );
    assert_eq!(
        repository.sessions["session-1"]
            .student()
            .unwrap()
            .number
            .as_str(),
        "04237384350"
    );
    assert!(repository.sessions["session-1"].needs_review());
}

#[test]
fn ambiguous_or_unknown_numbers_never_pick_a_student() {
    for number in ["00724475821", "99999999999"] {
        let (repository, page) = process_with_roster(number_only_extraction(number, 0.95));
        assert!(
            page.review_issues
                .iter()
                .any(|issue| issue.reason == ReviewReason::UnresolvedStudentIdentity),
            "{number}"
        );
        assert!(repository.sessions["session-1"].student().is_none());
    }
}
