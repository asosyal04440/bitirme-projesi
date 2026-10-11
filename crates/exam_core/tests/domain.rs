use exam_core::domain::*;

fn fixture() -> (Exam, StudentIdentity) {
    let outcome =
        LearningOutcome::new(LearningOutcomeId::new("LO-1").unwrap(), "Outcome 1").unwrap();
    let allocation = OutcomeAllocation::new(outcome.id.clone(), 10_000).unwrap();
    let question = Question::new(
        QuestionId::new("1").unwrap(),
        ScoreValue::from_decimal("10").unwrap(),
        [allocation],
    )
    .unwrap();
    let exam = Exam::new("exam-1", "Exam", [question], [outcome]).unwrap();
    let student = StudentIdentity::new("Ada", StudentNumber::new("123").unwrap()).unwrap();
    (exam, student)
}

fn result(page_index: u32, value: &str, confidence: f32) -> QuestionResult {
    result_for_question(page_index, "1", value, confidence)
}

fn result_for_question(
    page_index: u32,
    question_id: &str,
    value: &str,
    confidence: f32,
) -> QuestionResult {
    let bounds = BoundingBox::new(0.0, 0.0, 0.5, 0.5).unwrap();
    QuestionResult {
        question_id: QuestionId::new(question_id).unwrap(),
        score: Score::new(
            ScoreValue::from_decimal(value).unwrap(),
            ScoreValue::from_decimal("10").unwrap(),
        )
        .unwrap(),
        confidence: Confidence::new(confidence).unwrap(),
        source_region: SourceRegion::new(page_index, bounds),
    }
}

fn resolved_session(expected_pages: Option<u32>) -> ExamSession {
    let (exam, student) = fixture();
    let mut session = ExamSession::new("session-1", expected_pages).unwrap();
    session.merge_exam_discoveries(exam).unwrap();
    session.resolve_student(student).unwrap();
    session
}

#[test]
fn decimal_score_arithmetic_is_exact_without_floating_point() {
    let seven_point_five = ScoreValue::from_decimal("7.5").unwrap();
    let twelve_point_five = ScoreValue::from_decimal("12.50").unwrap();
    assert_eq!(
        seven_point_five
            .checked_add(twelve_point_five)
            .unwrap()
            .to_string(),
        "20"
    );
    assert_eq!(
        ScoreValue::from_decimal("0.25").unwrap().to_string(),
        "0.25"
    );
    assert!(ScoreValue::from_decimal("0.255").is_err());
}

#[test]
fn invalid_scores_cannot_exceed_question_maximum() {
    let value = ScoreValue::from_decimal("11").unwrap();
    let maximum = ScoreValue::from_decimal("10").unwrap();
    let error = Score::new(value, maximum).unwrap_err();
    assert_eq!(error, DomainError::ScoreExceedsMaximum { value, maximum });
}

#[test]
fn successful_pages_survive_a_failed_page() {
    let mut session = resolved_session(Some(3));
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, Some(0), Some(1)),
            vec![result(0, "8", 0.9)],
        ))
        .unwrap();
    session.record_failure(PageFailure::new(1, "temporary analyzer failure"));

    assert_eq!(session.pages().len(), 1);
    assert!(session.pages().contains_key(&0));
    assert_eq!(
        session.failures().get(&1).unwrap().reason,
        "temporary analyzer failure"
    );
    assert_eq!(
        session
            .missing_page_indexes()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn learning_outcome_aggregation_is_deterministic() {
    let mut session = resolved_session(Some(2));
    let outcome_id = LearningOutcomeId::new("LO-1").unwrap();
    let additional_question = Question::new(
        QuestionId::new("2").unwrap(),
        ScoreValue::from_decimal("10").unwrap(),
        [OutcomeAllocation::new(outcome_id.clone(), 10_000).unwrap()],
    )
    .unwrap();
    session
        .merge_exam_discoveries(
            Exam::new(
                "exam-1",
                "Exam",
                [additional_question],
                [LearningOutcome::new(outcome_id, "Outcome 1").unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(1, Some(1), Some(2)),
            vec![result(1, "7", 0.8)],
        ))
        .unwrap();
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, Some(0), Some(1)),
            vec![result_for_question(0, "2", "8", 0.9)],
        ))
        .unwrap();

    let aggregate = session.aggregate_learning_outcomes().unwrap();
    let value = aggregate
        .get(&LearningOutcomeId::new("LO-1").unwrap())
        .unwrap();
    assert_eq!(value.earned_score.to_string(), "15");
    assert_eq!(value.possible_score.to_string(), "20");
    assert_eq!(value.minimum_confidence.unwrap().value(), 0.8);
}

#[test]
fn exam_rejects_duplicate_question_ids_and_unknown_outcomes() {
    let qid = QuestionId::new("1").unwrap();
    let first = Question::new(qid.clone(), ScoreValue::from_decimal("5").unwrap(), []).unwrap();
    let second = Question::new(qid.clone(), ScoreValue::from_decimal("5").unwrap(), []).unwrap();
    assert_eq!(
        Exam::new("e", "Exam", [first, second], []),
        Err(DomainError::DuplicateQuestion(qid.clone()))
    );

    let missing = LearningOutcomeId::new("missing").unwrap();
    let question = Question::new(
        qid,
        ScoreValue::from_decimal("5").unwrap(),
        [OutcomeAllocation::new(missing.clone(), 10_000).unwrap()],
    )
    .unwrap();
    assert_eq!(
        Exam::new("e", "Exam", [question], []),
        Err(DomainError::UnknownLearningOutcome(missing))
    );
}

#[test]
fn exam_rejects_duplicate_learning_outcome_ids() {
    let first = LearningOutcome::new(LearningOutcomeId::new("LO-1").unwrap(), "First").unwrap();
    let second =
        LearningOutcome::new(LearningOutcomeId::new("LO-1").unwrap(), "Duplicate").unwrap();
    assert_eq!(
        Exam::new("e", "Exam", [], [first, second]),
        Err(DomainError::DuplicateLearningOutcome(
            LearningOutcomeId::new("LO-1").unwrap()
        ))
    );
}

#[test]
fn allocation_shares_prevent_multi_outcome_double_counting() {
    let outcome_a =
        LearningOutcome::new(LearningOutcomeId::new("A").unwrap(), "Outcome A").unwrap();
    let outcome_b =
        LearningOutcome::new(LearningOutcomeId::new("B").unwrap(), "Outcome B").unwrap();
    let question = Question::new(
        QuestionId::new("3").unwrap(),
        ScoreValue::from_decimal("20").unwrap(),
        [
            OutcomeAllocation::new(outcome_a.id.clone(), 5_000).unwrap(),
            OutcomeAllocation::new(outcome_b.id.clone(), 5_000).unwrap(),
        ],
    )
    .unwrap();
    let exam = Exam::new(
        "exam-1",
        "Exam",
        [question],
        [outcome_a.clone(), outcome_b.clone()],
    )
    .unwrap();
    let mut session = ExamSession::new("session-1", None).unwrap();
    session.merge_exam_discoveries(exam).unwrap();
    let score = Score::new(
        ScoreValue::from_decimal("20").unwrap(),
        ScoreValue::from_decimal("20").unwrap(),
    )
    .unwrap();
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, None, Some(1)),
            vec![QuestionResult {
                question_id: QuestionId::new("3").unwrap(),
                score,
                confidence: Confidence::new(1.0).unwrap(),
                source_region: SourceRegion::new(0, BoundingBox::new(0.0, 0.0, 0.5, 0.5).unwrap()),
            }],
        ))
        .unwrap();

    let totals = session.aggregate_learning_outcomes().unwrap();
    assert_eq!(totals[&outcome_a.id].earned_score.to_string(), "10");
    assert_eq!(totals[&outcome_b.id].earned_score.to_string(), "10");
    assert_eq!(totals[&outcome_a.id].possible_score.to_string(), "10");
    assert_eq!(totals[&outcome_b.id].possible_score.to_string(), "10");
}

#[test]
fn unresolved_allocation_cannot_produce_a_misleading_aggregate() {
    let (exam, student) = fixture();
    let unresolved_question = Question::new(
        QuestionId::new("2").unwrap(),
        ScoreValue::from_decimal("10").unwrap(),
        [],
    )
    .unwrap();
    let exam = Exam::new(
        exam.id,
        exam.title,
        exam.questions.into_values().chain([unresolved_question]),
        exam.learning_outcomes.into_values(),
    )
    .unwrap();
    let mut session = ExamSession::new("session-1", None).unwrap();
    session.merge_exam_discoveries(exam).unwrap();
    session.resolve_student(student).unwrap();
    session
        .checkpoint_page(
            ExamPage::new(PageMetadata::new(0, None, Some(1)), vec![]).with_review_issues(vec![
                ReviewIssue::new(
                    ReviewReason::AmbiguousOutcomeMapping,
                    "question 2 outcome mapping",
                    Some("A + B".to_owned()),
                    None,
                ),
            ]),
        )
        .unwrap();
    assert!(session.aggregate_learning_outcomes().unwrap().is_empty());
    assert!(session.needs_review());
}

#[test]
fn invalid_outcome_allocation_total_is_rejected() {
    let id = LearningOutcomeId::new("LO-1").unwrap();
    let allocation = OutcomeAllocation::new(id.clone(), 5_000).unwrap();
    assert_eq!(
        Question::new(
            QuestionId::new("1").unwrap(),
            ScoreValue::from_decimal("10").unwrap(),
            [allocation]
        ),
        Err(DomainError::InvalidOutcomeAllocationTotal {
            total_basis_points: 5_000
        })
    );
}

#[test]
fn checkpoint_rejects_a_source_region_from_another_page() {
    let mut session = resolved_session(None);
    let mut result = result(0, "5", 0.9);
    result.source_region.page_index = 1;
    assert_eq!(
        session.checkpoint_page(ExamPage::new(
            PageMetadata::new(0, None, None),
            vec![result],
        )),
        Err(DomainError::SourceRegionPageMismatch {
            expected: 0,
            actual: 1
        })
    );
}

#[test]
fn duplicate_page_is_rejected_and_success_clears_a_prior_failure() {
    let mut session = resolved_session(None);
    session.record_failure(PageFailure::new(0, "temporary failure"));
    let page = ExamPage::new(PageMetadata::new(0, None, None), vec![result(0, "5", 0.9)]);
    session.checkpoint_page(page.clone()).unwrap();
    assert!(!session.failures().contains_key(&0));
    assert_eq!(
        session.checkpoint_page(page),
        Err(DomainError::DuplicatePage(0))
    );
}

#[test]
fn session_can_begin_unresolved_then_resolve_student_and_exam() {
    let (exam, student) = fixture();
    let mut session = ExamSession::new("session-1", Some(1)).unwrap();
    assert!(session.exam().is_none());
    assert!(session.student().is_none());
    session.resolve_student(student).unwrap();
    session.merge_exam_discoveries(exam).unwrap();
    assert!(session.exam().is_some());
    assert!(session.student().is_some());
}

#[test]
fn question_ids_canonicalize_form_spellings_without_collapsing_inner_punctuation() {
    for spelling in [
        "1", "  S1 ", "s1", "(S1)", "S.1.", "S1:", "Soru 1", "SORU 1", "[1]",
    ] {
        assert_eq!(
            QuestionId::new(spelling).unwrap().as_str(),
            "1",
            "{spelling}"
        );
    }
    assert_eq!(QuestionId::new("Q3-b").unwrap().as_str(), "3-b");
    assert_eq!(
        QuestionId::new("q3-B").unwrap(),
        QuestionId::new("3-b").unwrap()
    );
    assert_ne!(
        QuestionId::new("1-a").unwrap(),
        QuestionId::new("1.1").unwrap()
    );
    assert_ne!(
        QuestionId::new("1.1").unwrap(),
        QuestionId::new("1a").unwrap()
    );
    assert_ne!(
        QuestionId::new("1-a").unwrap(),
        QuestionId::new("1a").unwrap()
    );
    // A prefix is removed only when a number follows it.
    for id in ["1-a", "1.1", "A", "S", "Sa", "Soru"] {
        assert_eq!(
            QuestionId::new(id).unwrap().as_str(),
            id.to_ascii_lowercase()
        );
    }
    assert!(QuestionId::new("()").is_err());
    assert!(QuestionId::new("  ").is_err());
}

#[test]
fn identical_question_observations_keep_provenance_but_aggregate_once() {
    let mut session = resolved_session(Some(2));
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, Some(0), Some(1)),
            vec![result(0, "8", 0.9)],
        ))
        .unwrap();
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(1, Some(1), Some(2)),
            vec![result(1, "8", 0.8)],
        ))
        .unwrap();

    let canonical = &session.canonical_question_results()[&QuestionId::new("1").unwrap()];
    assert_eq!(canonical.score.unwrap().value().to_string(), "8");
    assert!(!canonical.has_conflict);
    assert_eq!(canonical.observations.len(), 2);
    assert_eq!(canonical.observations[0].source_region.page_index, 0);
    assert_eq!(canonical.observations[1].source_region.page_index, 1);
    let aggregate = session.aggregate_learning_outcomes().unwrap();
    let result = aggregate.values().next().unwrap();
    assert_eq!(result.earned_score.to_string(), "8");
    assert_eq!(result.possible_score.to_string(), "10");
    assert_eq!(result.minimum_confidence.unwrap().value(), 0.8);
}

#[test]
fn conflicting_question_observations_are_both_preserved_and_excluded() {
    let mut session = resolved_session(Some(2));
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, Some(0), Some(1)),
            vec![result(0, "8", 0.9)],
        ))
        .unwrap();
    let second_page = ExamPage::new(
        PageMetadata::new(1, Some(1), Some(2)),
        vec![result(1, "7", 0.8)],
    );
    session.checkpoint_page(second_page).unwrap();

    let canonical = &session.canonical_question_results()[&QuestionId::new("1").unwrap()];
    assert_eq!(canonical.score, None);
    assert!(canonical.has_conflict);
    assert_eq!(canonical.observations.len(), 2);
    assert!(session.aggregate_learning_outcomes().unwrap().is_empty());
    assert!(session.needs_review());
    assert!(session.pages()[&1].review_issues.iter().any(|issue| {
        issue.reason == ReviewReason::ConflictingQuestionScore
            && issue
                .candidate
                .as_deref()
                .is_some_and(|text| text.contains("8 / 10") && text.contains("7 / 10"))
    }));
}

#[test]
fn conflicting_pending_question_definitions_remain_unresolved_for_review() {
    let question_id = QuestionId::new("Q3-b").unwrap();
    let mut session = ExamSession::new("session-1", Some(2)).unwrap();
    session.add_pending_question_definition(PendingQuestionDefinition::new(
        question_id.clone(),
        Some(ScoreValue::from_decimal("15").unwrap()),
        None,
    ));
    session.add_pending_question_definition(PendingQuestionDefinition::new(
        question_id.clone(),
        Some(ScoreValue::from_decimal("20").unwrap()),
        None,
    ));
    session.reconcile_pending_discoveries().unwrap();

    assert!(session.exam().is_none());
    assert_eq!(session.pending_questions().len(), 1);
    assert!(session.pending_questions()[&question_id]
        .resolved_maximum()
        .is_none());
    assert!(session.needs_review());
}

fn outcome(id: &str) -> LearningOutcomeId {
    LearningOutcomeId::new(id).unwrap()
}

fn question(id: &str) -> QuestionId {
    QuestionId::new(id).unwrap()
}

fn points(value: &str) -> ScoreValue {
    ScoreValue::from_decimal(value).unwrap()
}

/// Discoveries as printed on the İnönü form: `(S1)-PÇ2 (S2)-PÇ1`, no weights,
/// no outcome labels.
fn discover_form_table(session: &mut ExamSession) {
    for (question_id, outcome_id, maximum) in [("(S1)", "PÇ2", "20"), ("(S2)", "PÇ1", "30")] {
        session.add_pending_question_definition(PendingQuestionDefinition::new(
            question(question_id),
            Some(points(maximum)),
            None,
        ));
        session.add_pending_learning_outcome(PendingLearningOutcomeDefinition::new(
            outcome(outcome_id),
            None,
            None,
        ));
        session.add_pending_outcome_mapping(PendingOutcomeMapping::new(
            question(question_id),
            outcome(outcome_id),
            None,
            None,
        ));
    }
}

#[test]
fn outcomes_printed_without_labels_resolve_by_identifier() {
    let mut session = ExamSession::new("session-1", Some(1)).unwrap();
    discover_form_table(&mut session);
    session.reconcile_pending_discoveries().unwrap();

    let exam = session.exam().unwrap();
    assert_eq!(exam.learning_outcomes[&outcome("PÇ2")].label, None);
    assert!(session.pending_learning_outcomes().is_empty());
}

#[test]
fn unreadable_outcome_label_is_not_treated_as_unlabeled() {
    let mut session = ExamSession::new("session-1", Some(1)).unwrap();
    session.add_pending_learning_outcome(
        PendingLearningOutcomeDefinition::new(outcome("PÇ2"), None, None).with_unreadable_label(),
    );
    session.reconcile_pending_discoveries().unwrap();

    assert!(session.exam().is_none());
    assert_eq!(
        session.pending_learning_outcomes()[&outcome("PÇ2")].label_resolution(),
        LabelResolution::Unresolved
    );
    assert!(session.needs_review());
}

#[test]
fn a_later_label_fills_an_unlabeled_outcome_but_different_labels_conflict() {
    let unlabeled = Exam::new(
        "exam-1",
        "Exam",
        [],
        [LearningOutcome::unlabeled(outcome("PÇ2"))],
    )
    .unwrap();
    let labeled = Exam::new(
        "exam-1",
        "Exam",
        [],
        [LearningOutcome::new(outcome("PÇ2"), "Problem solving").unwrap()],
    )
    .unwrap();
    let merged = unlabeled.merge_discoveries(labeled.clone()).unwrap();
    assert_eq!(
        merged.learning_outcomes[&outcome("PÇ2")].label.as_deref(),
        Some("Problem solving")
    );
    assert_eq!(
        labeled
            .merge_discoveries(unlabeled)
            .unwrap()
            .learning_outcomes[&outcome("PÇ2")]
            .label
            .as_deref(),
        Some("Problem solving")
    );
    let other = Exam::new(
        "exam-1",
        "Exam",
        [],
        [LearningOutcome::new(outcome("PÇ2"), "Teamwork").unwrap()],
    )
    .unwrap();
    assert_eq!(
        merged.merge_discoveries(other),
        Err(DomainError::ConflictingLearningOutcome(outcome("PÇ2")))
    );
}

#[test]
fn unweighted_single_outcome_mapping_needs_the_explicit_form_rule() {
    let mut session = ExamSession::new("session-1", Some(1)).unwrap();
    discover_form_table(&mut session);
    session.reconcile_pending_discoveries().unwrap();

    let exam = session.exam().unwrap();
    assert!(exam.questions[&question("1")]
        .outcome_allocations
        .is_empty());
    assert_eq!(session.pending_outcome_mappings().len(), 2);
    assert!(session.needs_review());
}

#[test]
fn single_outcome_rule_gives_full_weight_and_aggregates_outcomes() {
    let mut session = ExamSession::new("session-1", Some(1))
        .unwrap()
        .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight);
    discover_form_table(&mut session);
    session.reconcile_pending_discoveries().unwrap();

    let exam = session.exam().unwrap();
    assert_eq!(
        exam.questions[&question("S.1.")].outcome_allocations,
        vec![OutcomeAllocation::new(outcome("PÇ2"), 10_000).unwrap()]
    );
    assert_eq!(session.pending_discovery_count(), 0);

    // The same printed table seen again on another page stays resolved.
    discover_form_table(&mut session);
    session.reconcile_pending_discoveries().unwrap();
    assert_eq!(session.pending_discovery_count(), 0);

    let bounds = BoundingBox::new(0.1, 0.1, 0.2, 0.2).unwrap();
    let score = |id: &str, value: &str, maximum: &str| QuestionResult {
        question_id: question(id),
        score: Score::new(points(value), points(maximum)).unwrap(),
        confidence: Confidence::new(0.9).unwrap(),
        source_region: SourceRegion::new(0, bounds),
    };
    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(0, Some(0), Some(1)),
            vec![score("S1", "15", "20"), score("S2", "21", "30")],
        ))
        .unwrap();
    let outcomes = session.aggregate_learning_outcomes().unwrap();
    assert_eq!(outcomes[&outcome("PÇ2")].earned_score, points("15"));
    assert_eq!(outcomes[&outcome("PÇ2")].possible_score, points("20"));
    assert_eq!(outcomes[&outcome("PÇ1")].earned_score, points("21"));
    assert_eq!(outcomes[&outcome("PÇ1")].possible_score, points("30"));
}

#[test]
fn single_outcome_rule_never_splits_or_overrides_observed_weights() {
    let mut session = ExamSession::new("session-1", Some(1))
        .unwrap()
        .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight);
    for id in ["1", "2"] {
        session.add_pending_question_definition(PendingQuestionDefinition::new(
            question(id),
            Some(points("20")),
            None,
        ));
    }
    for id in ["PÇ1", "PÇ2"] {
        session.add_pending_learning_outcome(PendingLearningOutcomeDefinition::new(
            outcome(id),
            None,
            None,
        ));
    }
    // Question 1 maps to two outcomes without weights: no equal split.
    for id in ["PÇ1", "PÇ2"] {
        session.add_pending_outcome_mapping(PendingOutcomeMapping::new(
            question("1"),
            outcome(id),
            None,
            None,
        ));
    }
    // Question 2 has a printed weight that could not be read.
    session.add_pending_outcome_mapping(
        PendingOutcomeMapping::new(question("2"), outcome("PÇ1"), None, None)
            .with_unreadable_allocation(),
    );
    session.reconcile_pending_discoveries().unwrap();

    let exam = session.exam().unwrap();
    assert!(exam.questions[&question("1")]
        .outcome_allocations
        .is_empty());
    assert!(exam.questions[&question("2")]
        .outcome_allocations
        .is_empty());
    assert_eq!(session.pending_outcome_mappings().len(), 3);
    assert!(session.needs_review());
}

fn declared_total(page_index: u32, value: &str) -> DeclaredTotal {
    DeclaredTotal {
        value: points(value),
        confidence: Confidence::new(0.9).unwrap(),
        source_region: SourceRegion::new(page_index, BoundingBox::new(0.6, 0.6, 0.1, 0.1).unwrap()),
    }
}

#[test]
fn declared_total_matching_the_scores_is_consistent() {
    let mut session = resolved_session(Some(1));
    session
        .checkpoint_page(
            ExamPage::new(
                PageMetadata::new(0, Some(0), Some(1)),
                vec![result(0, "7", 0.9)],
            )
            .with_declared_total(declared_total(0, "7.00")),
        )
        .unwrap();
    assert_eq!(session.total_check(), TotalCheck::Consistent);
    assert!(session.pages()[&0].review_issues.is_empty());
    assert!(!session.needs_review());
}

#[test]
fn declared_total_mismatch_is_flagged_on_the_page_and_session() {
    let mut session = resolved_session(Some(1));
    session
        .checkpoint_page(
            ExamPage::new(
                PageMetadata::new(0, Some(0), Some(1)),
                vec![result(0, "7", 0.9)],
            )
            .with_declared_total(declared_total(0, "8")),
        )
        .unwrap();
    assert_eq!(
        session.total_check(),
        TotalCheck::Mismatch {
            declared: points("8"),
            computed: points("7"),
        }
    );
    assert!(session.pages()[&0]
        .review_issues
        .iter()
        .any(|issue| issue.reason == ReviewReason::TotalMismatch));
    assert!(session.needs_review());
}

#[test]
fn declared_total_waits_for_scores_from_later_pages() {
    let mut session = resolved_session(Some(2));
    session
        .checkpoint_page(
            ExamPage::new(PageMetadata::new(0, Some(0), Some(1)), Vec::new())
                .with_declared_total(declared_total(0, "8")),
        )
        .unwrap();
    assert_eq!(session.total_check(), TotalCheck::Incomplete);
    assert!(session.pages()[&0].review_issues.is_empty());

    session
        .checkpoint_page(ExamPage::new(
            PageMetadata::new(1, Some(1), Some(2)),
            vec![result(1, "7", 0.9)],
        ))
        .unwrap();
    assert_eq!(
        session.total_check(),
        TotalCheck::Mismatch {
            declared: points("8"),
            computed: points("7"),
        }
    );
    assert!(session.needs_review());
}
