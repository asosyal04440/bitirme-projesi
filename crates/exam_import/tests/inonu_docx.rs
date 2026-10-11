//! Import of the sample exam file (the real İnönü sheet with the instructor name and question
//! texts removed).

use exam_core::domain::{LearningOutcomeId, QuestionId, ScoreValue};
use exam_import::{ColumnSpan, ImportWarning, ImportedExam};

fn sample() -> ImportedExam {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/synthetic/templates/inonu-final-a.docx"
    ))
    .unwrap();
    ImportedExam::from_docx(&bytes).unwrap()
}

fn id(text: &str) -> QuestionId {
    QuestionId::new(text).unwrap()
}

#[test]
fn header_metadata_is_read() {
    let meta = sample().metadata;
    assert_eq!(meta.academic_year.as_deref(), Some("2025-2026"));
    assert_eq!(meta.term.as_deref(), Some("Bahar"));
    assert_eq!(meta.exam_kind.as_deref(), Some("Final"));
    assert_eq!(meta.course.as_deref(), Some("BİLM374-2022-YAPAY ZEKA"));
    assert_eq!(meta.course_code.as_deref(), Some("BİLM374"));
    assert_eq!(meta.course_name.as_deref(), Some("YAPAY ZEKA"));
    assert_eq!(meta.department.as_deref(), Some("Bilgisayar Mühendisliği"));
    assert_eq!(meta.group.as_deref(), Some("A"));
    assert!(meta
        .date
        .as_deref()
        .is_some_and(|d| d.starts_with("10/04/2026")));
}

#[test]
fn question_table_and_points_are_read_and_gaps_reported() {
    let exam = sample();
    let pairs: Vec<(&str, Vec<&str>)> = exam
        .questions
        .iter()
        .map(|q| {
            (
                q.printed_id.as_str(),
                q.outcomes.iter().map(LearningOutcomeId::as_str).collect(),
            )
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("S1", vec!["PÇ2"]),
            ("S2", vec!["PÇ1"]),
            ("S3", vec!["PÇ3"]),
            ("S4", vec!["PÇ2"])
        ]
    );
    let maxima: Vec<Option<String>> = exam
        .questions
        .iter()
        .map(|q| q.maximum_score.map(|v| v.to_string()))
        .collect();
    assert_eq!(maxima, [Some("20".into()), Some("30".into()), None, None]);
    assert_eq!(
        exam.questions[1].maximum_source.as_deref(),
        Some("(3*10=30 puan)")
    );
    // The sample file only contains the texts of S1 and S2.
    assert_eq!(
        exam.warnings,
        vec![
            ImportWarning::MissingMaximumScore(id("3")),
            ImportWarning::MissingMaximumScore(id("4"))
        ]
    );
}

#[test]
fn score_cells_come_from_the_word_column_widths() {
    let exam = sample();
    // Left margin 709 twips, label column 1336, question columns 1121/1121/1121/1108.
    let columns: Vec<ColumnSpan> = exam.questions.iter().map(|q| q.score_column).collect();
    assert_eq!(
        columns,
        [(2045, 1121), (3166, 1121), (4287, 1121), (5408, 1108)].map(|(left, width)| ColumnSpan {
            left_twips: left,
            width_twips: width
        })
    );
    assert_eq!(
        exam.total_column,
        ColumnSpan {
            left_twips: 2045,
            width_twips: 4471
        }
    );
    assert_eq!(exam.page_width_twips, 12240);
}

#[test]
fn completed_import_becomes_a_domain_exam() {
    let incomplete = sample();
    assert!(incomplete
        .to_exam("yz-final-a", "Yapay Zeka Final A")
        .is_err());
    let twenty_five = ScoreValue::from_decimal("25").unwrap();
    let exam = incomplete
        .with_maximum_score(&id("S3"), twenty_five)
        .with_maximum_score(&id("S4"), twenty_five);
    assert!(exam.warnings.is_empty());
    let domain = exam.to_exam("yz-final-a", "Yapay Zeka Final A").unwrap();
    assert_eq!(domain.questions.len(), 4);
    assert_eq!(domain.learning_outcomes.len(), 3);
    let question = &domain.questions[&id("1")];
    assert_eq!(question.maximum_score.to_string(), "20");
    assert_eq!(question.outcome_allocations[0].weight_basis_points, 10_000);
    assert!(domain.learning_outcomes.values().all(|o| o.label.is_none()));
}
