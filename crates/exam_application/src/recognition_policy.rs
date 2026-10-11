use exam_core::domain::Confidence;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecognitionFieldKind {
    StudentName,
    StudentNumber,
    QuestionId,
    QuestionMaximum,
    Score,
    LearningOutcomeId,
    LearningOutcomeLabel,
    OutcomeMapping,
    OutcomeAllocation,
    DeclaredTotal,
}

impl RecognitionFieldKind {
    const ALL: [Self; 10] = [
        Self::StudentName,
        Self::StudentNumber,
        Self::QuestionId,
        Self::QuestionMaximum,
        Self::Score,
        Self::LearningOutcomeId,
        Self::LearningOutcomeLabel,
        Self::OutcomeMapping,
        Self::OutcomeAllocation,
        Self::DeclaredTotal,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecognitionPolicyError;

impl fmt::Display for RecognitionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("recognition confidence threshold must be finite and between 0 and 1")
    }
}

impl std::error::Error for RecognitionPolicyError {}

/// Application-level confidence gates; semantic domain validation remains separate.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionPolicy {
    minimum_confidence: BTreeMap<RecognitionFieldKind, f32>,
}

impl Default for RecognitionPolicy {
    fn default() -> Self {
        Self {
            minimum_confidence: RecognitionFieldKind::ALL
                .into_iter()
                .map(|kind| (kind, 0.8))
                .collect(),
        }
    }
}

impl RecognitionPolicy {
    pub fn with_minimum_confidence(
        mut self,
        kind: RecognitionFieldKind,
        threshold: f32,
    ) -> Result<Self, RecognitionPolicyError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(RecognitionPolicyError);
        }
        self.minimum_confidence.insert(kind, threshold);
        Ok(self)
    }

    pub fn minimum_confidence(&self, kind: RecognitionFieldKind) -> f32 {
        self.minimum_confidence.get(&kind).copied().unwrap_or(0.8)
    }

    pub fn accepts(&self, kind: RecognitionFieldKind, confidence: Option<Confidence>) -> bool {
        confidence.is_some_and(|confidence| confidence.value() >= self.minimum_confidence(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn confidence(value: f32) -> Confidence {
        Confidence::new(value).unwrap()
    }

    #[test]
    fn default_threshold_is_field_specific_and_inclusive() {
        let policy = RecognitionPolicy::default();
        let fields = [
            RecognitionFieldKind::StudentName,
            RecognitionFieldKind::StudentNumber,
            RecognitionFieldKind::QuestionId,
            RecognitionFieldKind::QuestionMaximum,
            RecognitionFieldKind::Score,
            RecognitionFieldKind::LearningOutcomeId,
            RecognitionFieldKind::LearningOutcomeLabel,
            RecognitionFieldKind::OutcomeMapping,
            RecognitionFieldKind::OutcomeAllocation,
            RecognitionFieldKind::DeclaredTotal,
        ];

        for field in fields {
            assert_eq!(policy.minimum_confidence(field), 0.8);
            assert!(policy.accepts(field, Some(confidence(0.8))));
            assert!(!policy.accepts(field, Some(confidence(0.79))));
            assert!(!policy.accepts(field, None));
        }
    }

    #[test]
    fn threshold_override_only_changes_the_selected_field() {
        let policy = RecognitionPolicy::default()
            .with_minimum_confidence(RecognitionFieldKind::Score, 0.95)
            .unwrap();

        assert_eq!(policy.minimum_confidence(RecognitionFieldKind::Score), 0.95);
        assert_eq!(
            policy.minimum_confidence(RecognitionFieldKind::StudentName),
            0.8
        );
        assert!(policy.accepts(RecognitionFieldKind::Score, Some(confidence(0.95))));
        assert!(!policy.accepts(RecognitionFieldKind::Score, Some(confidence(0.94))));
    }

    #[test]
    fn thresholds_outside_the_probability_interval_are_rejected() {
        for threshold in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            assert!(RecognitionPolicy::default()
                .with_minimum_confidence(RecognitionFieldKind::Score, threshold)
                .is_err());
        }
        assert!(RecognitionPolicy::default()
            .with_minimum_confidence(RecognitionFieldKind::Score, 0.0)
            .is_ok());
        assert!(RecognitionPolicy::default()
            .with_minimum_confidence(RecognitionFieldKind::Score, 1.0)
            .is_ok());
    }
}
