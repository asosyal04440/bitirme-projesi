use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainError {
    EmptyValue(&'static str),
    InvalidValue(&'static str),
    ScoreExceedsMaximum {
        value: ScoreValue,
        maximum: ScoreValue,
    },
    ScoreOverflow,
    InexactAllocation,
    DuplicateQuestion(QuestionId),
    DuplicateLearningOutcome(LearningOutcomeId),
    DuplicateOutcomeAllocation(LearningOutcomeId),
    InvalidOutcomeAllocationTotal {
        total_basis_points: u32,
    },
    UnknownLearningOutcome(LearningOutcomeId),
    ConflictingQuestionDefinition(QuestionId),
    ConflictingQuestionScore(QuestionId),
    ConflictingLearningOutcome(LearningOutcomeId),
    ConflictingStudentIdentity,
    UnknownQuestion(QuestionId),
    QuestionMaximumMismatch {
        question: QuestionId,
        expected: ScoreValue,
        actual: ScoreValue,
    },
    PageIndexMismatch {
        expected: u32,
        actual: u32,
    },
    SourceRegionPageMismatch {
        expected: u32,
        actual: u32,
    },
    DuplicatePage(u32),
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyValue(field) => write!(f, "{field} must not be empty"),
            Self::InvalidValue(field) => write!(f, "{field} has an invalid value"),
            Self::ScoreExceedsMaximum { value, maximum } => {
                write!(f, "score {value} exceeds maximum {maximum}")
            }
            Self::ScoreOverflow => write!(f, "score arithmetic overflow"),
            Self::InexactAllocation => write!(f, "outcome allocation is not exactly representable"),
            Self::DuplicateQuestion(id) => write!(f, "duplicate question {id}"),
            Self::DuplicateLearningOutcome(id) => {
                write!(f, "duplicate learning outcome {}", id.as_str())
            }
            Self::DuplicateOutcomeAllocation(id) => {
                write!(f, "duplicate outcome allocation {}", id.as_str())
            }
            Self::InvalidOutcomeAllocationTotal { total_basis_points } => write!(
                f,
                "outcome allocations total {total_basis_points} basis points, expected 10000"
            ),
            Self::UnknownLearningOutcome(id) => {
                write!(f, "unknown learning outcome {}", id.as_str())
            }
            Self::ConflictingQuestionDefinition(id) => {
                write!(f, "conflicting definition for question {id}")
            }
            Self::ConflictingQuestionScore(id) => {
                write!(f, "conflicting score observations for question {id}")
            }
            Self::ConflictingLearningOutcome(id) => {
                write!(
                    f,
                    "conflicting definition for learning outcome {}",
                    id.as_str()
                )
            }
            Self::ConflictingStudentIdentity => {
                write!(f, "conflicting student identity in exam session")
            }
            Self::UnknownQuestion(id) => write!(f, "unknown question {id}"),
            Self::QuestionMaximumMismatch {
                question,
                expected,
                actual,
            } => write!(
                f,
                "question {} expects maximum {expected}, received {actual}",
                question
            ),
            Self::PageIndexMismatch { expected, actual } => {
                write!(
                    f,
                    "page index mismatch: expected {expected}, received {actual}"
                )
            }
            Self::SourceRegionPageMismatch { expected, actual } => write!(
                f,
                "source region belongs to page {actual}, expected page {expected}"
            ),
            Self::DuplicatePage(index) => write!(f, "page {index} is already checkpointed"),
        }
    }
}

impl std::error::Error for DomainError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuestionId(String);

impl QuestionId {
    const PREFIXES: [&'static str; 4] = ["question", "soru", "q", "s"];

    /// Question identifiers trim surrounding whitespace and ASCII-case-fold.
    /// Wrapping brackets, trailing `.`/`:` and a question prefix (`S`, `Soru`,
    /// `Q`, `Question`) directly followed by a digit are removed, so `(S1)`,
    /// `S.1.` and `Soru 1` all become `1`. Inner punctuation is kept: `1-a`,
    /// `1.1` and `1a` stay distinct.
    pub fn new(value: impl Into<String>) -> Result<Self, DomainError> {
        let value = value.into().trim().to_ascii_lowercase();
        let mut core = value.as_str();
        loop {
            let stripped = core
                .trim()
                .trim_start_matches(['(', '['])
                .trim_end_matches([')', ']', '.', ':']);
            if stripped == core {
                break;
            }
            core = stripped;
        }
        for prefix in Self::PREFIXES {
            if let Some(rest) = core.strip_prefix(prefix) {
                let rest = rest.trim_start_matches([' ', '.', ':']);
                if rest.starts_with(|c: char| c.is_ascii_digit()) {
                    core = rest;
                    break;
                }
            }
        }
        if core.is_empty() {
            return Err(DomainError::EmptyValue("question id"));
        }
        Ok(Self(core.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for QuestionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LearningOutcomeId(String);

impl LearningOutcomeId {
    pub fn new(value: impl Into<String>) -> Result<Self, DomainError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DomainError::EmptyValue("learning outcome id"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LearningOutcomeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StudentNumber(String);

impl StudentNumber {
    pub fn new(value: impl Into<String>) -> Result<Self, DomainError> {
        let value = value.into();
        let normalized = value.trim();
        if normalized.is_empty() || normalized.len() > 64 {
            return Err(DomainError::InvalidValue("student number"));
        }
        Ok(Self(normalized.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudentIdentity {
    pub name: String,
    pub number: StudentNumber,
}

impl StudentIdentity {
    pub fn new(name: impl Into<String>, number: StudentNumber) -> Result<Self, DomainError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(DomainError::EmptyValue("student name"));
        }
        Ok(Self { name, number })
    }
}

/// Exact score arithmetic. Inputs are decimal points with at most two places;
/// internal micro-points also represent basis-point outcome allocations exactly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScoreValue(u64);

impl ScoreValue {
    const MICRO_POINTS_PER_POINT: u64 = 1_000_000;
    const MICRO_POINTS_PER_HUNDREDTH: u64 = 10_000;
    const ALLOCATION_DENOMINATOR: u128 = 10_000;

    pub const ZERO: Self = Self(0);

    pub fn from_hundredths(value: u64) -> Result<Self, DomainError> {
        value
            .checked_mul(Self::MICRO_POINTS_PER_HUNDREDTH)
            .map(Self)
            .ok_or(DomainError::ScoreOverflow)
    }

    pub fn from_decimal(value: &str) -> Result<Self, DomainError> {
        let value = value.trim();
        let mut parts = value.split('.');
        let whole = parts.next().unwrap_or_default();
        let fractional = parts.next();
        if parts.next().is_some() || whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(DomainError::InvalidValue("score"));
        }
        let whole = whole
            .parse::<u64>()
            .map_err(|_| DomainError::ScoreOverflow)?;
        let hundredths = match fractional {
            None | Some("") => 0,
            Some(value) if value.len() == 1 && value.bytes().all(|b| b.is_ascii_digit()) => {
                u64::from(value.as_bytes()[0] - b'0') * 10
            }
            Some(value) if value.len() == 2 && value.bytes().all(|b| b.is_ascii_digit()) => {
                u64::from(
                    value
                        .parse::<u8>()
                        .map_err(|_| DomainError::InvalidValue("score"))?,
                )
            }
            _ => return Err(DomainError::InvalidValue("score")),
        };
        whole
            .checked_mul(100)
            .and_then(|value| value.checked_add(hundredths))
            .ok_or(DomainError::ScoreOverflow)
            .and_then(Self::from_hundredths)
    }

    pub const fn micro_points(self) -> u64 {
        self.0
    }

    pub fn checked_add(self, other: Self) -> Result<Self, DomainError> {
        self.0
            .checked_add(other.0)
            .map(Self)
            .ok_or(DomainError::ScoreOverflow)
    }

    fn allocated(self, weight_basis_points: u16) -> Result<Self, DomainError> {
        let product = u128::from(self.0) * u128::from(weight_basis_points);
        if product % Self::ALLOCATION_DENOMINATOR != 0 {
            return Err(DomainError::InexactAllocation);
        }
        u64::try_from(product / Self::ALLOCATION_DENOMINATOR)
            .map(Self)
            .map_err(|_| DomainError::ScoreOverflow)
    }
}

impl fmt::Display for ScoreValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = self.0 / Self::MICRO_POINTS_PER_POINT;
        let fractional = self.0 % Self::MICRO_POINTS_PER_POINT;
        if fractional == 0 {
            return write!(f, "{whole}");
        }
        let mut digits = format!("{fractional:06}");
        while digits.ends_with('0') {
            digits.pop();
        }
        write!(f, "{whole}.{digits}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Score {
    value: ScoreValue,
    maximum: ScoreValue,
}

impl Score {
    pub fn new(value: ScoreValue, maximum: ScoreValue) -> Result<Self, DomainError> {
        if maximum == ScoreValue::ZERO {
            return Err(DomainError::InvalidValue("score maximum"));
        }
        if value > maximum {
            return Err(DomainError::ScoreExceedsMaximum { value, maximum });
        }
        Ok(Self { value, maximum })
    }

    pub const fn value(&self) -> ScoreValue {
        self.value
    }

    pub const fn maximum(&self) -> ScoreValue {
        self.maximum
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence(f32);

impl Confidence {
    pub fn new(value: f32) -> Result<Self, DomainError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(DomainError::InvalidValue("confidence"));
        }
        Ok(Self(value))
    }

    pub const fn value(&self) -> f32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    /// Constructs a bounding box in normalized image coordinates in [0, 1].
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, DomainError> {
        let valid = [x, y, width, height]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value));
        if !valid || width == 0.0 || height == 0.0 || x + width > 1.0 || y + height > 1.0 {
            return Err(DomainError::InvalidValue("bounding box"));
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceRegion {
    pub page_index: u32,
    pub bounds: BoundingBox,
}

impl SourceRegion {
    pub const fn new(page_index: u32, bounds: BoundingBox) -> Self {
        Self { page_index, bounds }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelVersion {
    pub name: String,
    pub version: String,
    pub configuration_version: Option<String>,
}

impl ModelVersion {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Result<Self, DomainError> {
        let name = name.into();
        let version = version.into();
        if name.trim().is_empty() || version.trim().is_empty() {
            return Err(DomainError::EmptyValue("model version"));
        }
        Ok(Self {
            name,
            version,
            configuration_version: None,
        })
    }

    pub fn with_configuration_version(
        mut self,
        version: impl Into<String>,
    ) -> Result<Self, DomainError> {
        let version = version.into();
        if version.trim().is_empty() {
            return Err(DomainError::EmptyValue("model configuration version"));
        }
        self.configuration_version = Some(version);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearningOutcome {
    pub id: LearningOutcomeId,
    /// Forms often print only the identifier (e.g. `PÇ2`); `None` means the
    /// source has no label, not that a label failed to be read.
    pub label: Option<String>,
}

impl LearningOutcome {
    pub fn new(id: LearningOutcomeId, label: impl Into<String>) -> Result<Self, DomainError> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(DomainError::EmptyValue("learning outcome label"));
        }
        Ok(Self {
            id,
            label: Some(label),
        })
    }

    pub fn unlabeled(id: LearningOutcomeId) -> Self {
        Self { id, label: None }
    }

    /// Combines two definitions of the same outcome. A label fills a missing
    /// one; two different labels conflict.
    fn merged_with(&self, other: &Self) -> Option<Self> {
        match (&self.label, &other.label) {
            (Some(left), Some(right)) if left != right => None,
            (None, Some(_)) => Some(other.clone()),
            _ => Some(self.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub id: QuestionId,
    pub maximum_score: ScoreValue,
    pub outcome_allocations: Vec<OutcomeAllocation>,
}

impl Question {
    pub fn new(
        id: QuestionId,
        maximum_score: ScoreValue,
        outcome_allocations: impl IntoIterator<Item = OutcomeAllocation>,
    ) -> Result<Self, DomainError> {
        if maximum_score == ScoreValue::ZERO {
            return Err(DomainError::InvalidValue("question maximum score"));
        }
        let outcome_allocations: Vec<_> = outcome_allocations.into_iter().collect();
        validate_allocations(&outcome_allocations)?;
        Ok(Self {
            id,
            maximum_score,
            outcome_allocations,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeAllocation {
    pub outcome_id: LearningOutcomeId,
    pub weight_basis_points: u16,
}

impl OutcomeAllocation {
    pub const TOTAL_BASIS_POINTS: u32 = 10_000;

    pub fn new(
        outcome_id: LearningOutcomeId,
        weight_basis_points: u16,
    ) -> Result<Self, DomainError> {
        if weight_basis_points == 0 || u32::from(weight_basis_points) > Self::TOTAL_BASIS_POINTS {
            return Err(DomainError::InvalidValue("outcome allocation weight"));
        }
        Ok(Self {
            outcome_id,
            weight_basis_points,
        })
    }
}

const FULL_WEIGHT_BASIS_POINTS: u16 = 10_000;

fn validate_allocations(allocations: &[OutcomeAllocation]) -> Result<(), DomainError> {
    let mut ids = BTreeSet::new();
    let mut total = 0_u32;
    for allocation in allocations {
        if !ids.insert(allocation.outcome_id.clone()) {
            return Err(DomainError::DuplicateOutcomeAllocation(
                allocation.outcome_id.clone(),
            ));
        }
        total += u32::from(allocation.weight_basis_points);
    }
    if !allocations.is_empty() && total != OutcomeAllocation::TOTAL_BASIS_POINTS {
        return Err(DomainError::InvalidOutcomeAllocationTotal {
            total_basis_points: total,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exam {
    pub id: String,
    pub title: String,
    pub questions: BTreeMap<QuestionId, Question>,
    pub learning_outcomes: BTreeMap<LearningOutcomeId, LearningOutcome>,
}

impl Exam {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        questions: impl IntoIterator<Item = Question>,
        learning_outcomes: impl IntoIterator<Item = LearningOutcome>,
    ) -> Result<Self, DomainError> {
        let id = id.into();
        let title = title.into();
        if id.trim().is_empty() {
            return Err(DomainError::EmptyValue("exam id"));
        }
        if title.trim().is_empty() {
            return Err(DomainError::EmptyValue("exam title"));
        }
        let mut question_map = BTreeMap::new();
        for question in questions {
            let id = question.id.clone();
            if question_map.insert(id.clone(), question).is_some() {
                return Err(DomainError::DuplicateQuestion(id));
            }
        }
        let mut outcome_map = BTreeMap::new();
        for outcome in learning_outcomes {
            let id = outcome.id.clone();
            if outcome_map.insert(id.clone(), outcome).is_some() {
                return Err(DomainError::DuplicateLearningOutcome(id));
            }
        }
        for question in question_map.values() {
            for allocation in &question.outcome_allocations {
                if !outcome_map.contains_key(&allocation.outcome_id) {
                    return Err(DomainError::UnknownLearningOutcome(
                        allocation.outcome_id.clone(),
                    ));
                }
            }
        }
        Ok(Self {
            id,
            title,
            questions: question_map,
            learning_outcomes: outcome_map,
        })
    }

    pub fn merge_discoveries(&self, discovered: Self) -> Result<Self, DomainError> {
        if self.id != discovered.id {
            return Err(DomainError::InvalidValue("exam identity"));
        }
        let mut outcomes = self.learning_outcomes.clone();
        for (id, outcome) in discovered.learning_outcomes {
            let merged = match outcomes.get(&id) {
                Some(existing) => existing
                    .merged_with(&outcome)
                    .ok_or_else(|| DomainError::ConflictingLearningOutcome(id.clone()))?,
                None => outcome,
            };
            outcomes.insert(id, merged);
        }

        let mut questions = self.questions.clone();
        for (id, question) in discovered.questions {
            if let Some(existing) = questions.get_mut(&id) {
                if existing.maximum_score != question.maximum_score {
                    return Err(DomainError::ConflictingQuestionDefinition(id));
                }
                if existing.outcome_allocations.is_empty() {
                    existing.outcome_allocations = question.outcome_allocations;
                } else if !question.outcome_allocations.is_empty()
                    && existing.outcome_allocations != question.outcome_allocations
                {
                    return Err(DomainError::ConflictingQuestionDefinition(id));
                }
            } else {
                questions.insert(id, question);
            }
        }

        Self::new(
            self.id.clone(),
            self.title.clone(),
            questions.into_values(),
            outcomes.into_values(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageOrderStatus {
    Unknown,
    InOrder,
    Unexpected,
    Duplicate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageMetadata {
    pub page_index: u32,
    pub expected_page_index: Option<u32>,
    pub source_page_number: Option<u32>,
    pub order_status: PageOrderStatus,
}

impl PageMetadata {
    pub fn new(
        page_index: u32,
        expected_page_index: Option<u32>,
        source_page_number: Option<u32>,
    ) -> Self {
        let order_status = match expected_page_index {
            None => PageOrderStatus::Unknown,
            Some(expected) if expected == page_index => PageOrderStatus::InOrder,
            Some(_) => PageOrderStatus::Unexpected,
        };
        Self {
            page_index,
            expected_page_index,
            source_page_number,
            order_status,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuestionResult {
    pub question_id: QuestionId,
    pub score: Score,
    pub confidence: Confidence,
    pub source_region: SourceRegion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewReason {
    LowConfidence,
    ScoreExceedsMaximum,
    UnknownQuestion,
    UnresolvedStudentIdentity,
    AmbiguousOutcomeMapping,
    AmbiguousCorrectedScore,
    InconsistentPageIdentity,
    SourceRegionMismatch,
    ConflictingQuestionScore,
    InvalidCandidate,
    TotalMismatch,
    /// A field that should be filled in was left empty on the paper.
    BlankField,
    /// The recognised student number was one digit away from exactly one
    /// enrolled student; that student is proposed, not assumed.
    RosterNearMatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReviewIssue {
    pub reason: ReviewReason,
    pub field: String,
    pub candidate: Option<String>,
    pub corrected_value: Option<String>,
    pub source_region: Option<SourceRegion>,
}

impl ReviewIssue {
    pub fn new(
        reason: ReviewReason,
        field: impl Into<String>,
        candidate: Option<String>,
        source_region: Option<SourceRegion>,
    ) -> Self {
        Self {
            reason,
            field: field.into(),
            candidate,
            corrected_value: None,
            source_region,
        }
    }

    pub fn with_correction(mut self, corrected_value: impl Into<String>) -> Self {
        self.corrected_value = Some(corrected_value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingQuestionDefinition {
    pub question_id: QuestionId,
    pub maximum_score_candidates: BTreeSet<ScoreValue>,
    pub source_regions: Vec<SourceRegion>,
}

impl PendingQuestionDefinition {
    pub fn new(
        question_id: QuestionId,
        maximum_score: Option<ScoreValue>,
        source_region: Option<SourceRegion>,
    ) -> Self {
        let mut maximum_score_candidates = BTreeSet::new();
        if let Some(maximum_score) = maximum_score {
            maximum_score_candidates.insert(maximum_score);
        }
        let mut pending = Self {
            question_id,
            maximum_score_candidates,
            source_regions: Vec::new(),
        };
        if let Some(source_region) = source_region {
            pending.add_source_region(source_region);
        }
        pending
    }

    pub fn add_candidate(&mut self, maximum_score: Option<ScoreValue>) {
        if let Some(maximum_score) = maximum_score {
            self.maximum_score_candidates.insert(maximum_score);
        }
    }

    pub fn add_source_region(&mut self, source_region: SourceRegion) {
        if !self.source_regions.contains(&source_region) {
            self.source_regions.push(source_region);
        }
    }

    pub fn resolved_maximum(&self) -> Option<ScoreValue> {
        (self.maximum_score_candidates.len() == 1).then(|| {
            *self
                .maximum_score_candidates
                .first()
                .expect("one candidate")
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelResolution<'a> {
    Labeled(&'a str),
    /// No label was observed anywhere; the outcome is identified by id only.
    Unlabeled,
    /// Conflicting labels, or a label was seen but could not be accepted.
    Unresolved,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingLearningOutcomeDefinition {
    pub outcome_id: LearningOutcomeId,
    pub label_candidates: BTreeSet<String>,
    /// A label was present on the page but rejected (e.g. low confidence), so
    /// the outcome must not be treated as unlabeled.
    pub unreadable_label_observed: bool,
    pub source_regions: Vec<SourceRegion>,
}

impl PendingLearningOutcomeDefinition {
    pub fn new(
        outcome_id: LearningOutcomeId,
        label: Option<String>,
        source_region: Option<SourceRegion>,
    ) -> Self {
        let mut label_candidates = BTreeSet::new();
        if let Some(label) = label.filter(|label| !label.trim().is_empty()) {
            label_candidates.insert(label.trim().to_owned());
        }
        let mut pending = Self {
            outcome_id,
            label_candidates,
            unreadable_label_observed: false,
            source_regions: Vec::new(),
        };
        if let Some(source_region) = source_region {
            pending.add_source_region(source_region);
        }
        pending
    }

    pub fn with_unreadable_label(mut self) -> Self {
        self.unreadable_label_observed = true;
        self
    }

    pub fn add_candidate(&mut self, label: Option<String>) {
        if let Some(label) = label.filter(|label| !label.trim().is_empty()) {
            self.label_candidates.insert(label.trim().to_owned());
        }
    }

    pub fn label_resolution(&self) -> LabelResolution<'_> {
        match (self.label_candidates.len(), self.unreadable_label_observed) {
            (0, false) => LabelResolution::Unlabeled,
            (1, _) => LabelResolution::Labeled(
                self.label_candidates
                    .first()
                    .map(String::as_str)
                    .expect("one candidate"),
            ),
            _ => LabelResolution::Unresolved,
        }
    }

    pub fn add_source_region(&mut self, source_region: SourceRegion) {
        if !self.source_regions.contains(&source_region) {
            self.source_regions.push(source_region);
        }
    }

    pub fn resolved_label(&self) -> Option<&str> {
        (self.label_candidates.len() == 1)
            .then(|| self.label_candidates.first().map(String::as_str))
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingOutcomeMapping {
    pub question_id: QuestionId,
    pub outcome_id: LearningOutcomeId,
    pub allocation_candidates: BTreeSet<u16>,
    /// A weight was present on the page but rejected, so an implied weight
    /// must not replace it.
    pub unreadable_allocation_observed: bool,
    pub source_regions: Vec<SourceRegion>,
}

impl PendingOutcomeMapping {
    pub fn new(
        question_id: QuestionId,
        outcome_id: LearningOutcomeId,
        allocation: Option<u16>,
        source_region: Option<SourceRegion>,
    ) -> Self {
        let mut allocation_candidates = BTreeSet::new();
        if let Some(allocation) = allocation {
            allocation_candidates.insert(allocation);
        }
        let mut pending = Self {
            question_id,
            outcome_id,
            allocation_candidates,
            unreadable_allocation_observed: false,
            source_regions: Vec::new(),
        };
        if let Some(source_region) = source_region {
            pending.add_source_region(source_region);
        }
        pending
    }

    pub fn with_unreadable_allocation(mut self) -> Self {
        self.unreadable_allocation_observed = true;
        self
    }

    /// True when no weight was printed for this mapping at all.
    fn has_no_allocation_evidence(&self) -> bool {
        self.allocation_candidates.is_empty() && !self.unreadable_allocation_observed
    }

    pub fn add_candidate(&mut self, allocation: Option<u16>) {
        if let Some(allocation) = allocation {
            self.allocation_candidates.insert(allocation);
        }
    }

    pub fn add_source_region(&mut self, source_region: SourceRegion) {
        if !self.source_regions.contains(&source_region) {
            self.source_regions.push(source_region);
        }
    }

    pub fn resolved_allocation(&self) -> Option<u16> {
        (self.allocation_candidates.len() == 1)
            .then(|| *self.allocation_candidates.first().expect("one candidate"))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PendingStudentIdentity {
    pub name_candidates: BTreeSet<String>,
    pub number_candidates: BTreeSet<StudentNumber>,
    pub source_regions: Vec<SourceRegion>,
}

impl PendingStudentIdentity {
    pub fn add_name(&mut self, name: Option<String>, source_region: Option<SourceRegion>) {
        if let Some(name) = name.filter(|name| !name.trim().is_empty()) {
            self.name_candidates.insert(name.trim().to_owned());
        }
        self.add_source_region(source_region);
    }

    pub fn add_number(
        &mut self,
        number: Option<StudentNumber>,
        source_region: Option<SourceRegion>,
    ) {
        if let Some(number) = number {
            self.number_candidates.insert(number);
        }
        self.add_source_region(source_region);
    }

    pub fn add_source_region(&mut self, source_region: Option<SourceRegion>) {
        if let Some(source_region) = source_region {
            if !self.source_regions.contains(&source_region) {
                self.source_regions.push(source_region);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.name_candidates.is_empty() && self.number_candidates.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalQuestionResult {
    pub question_id: QuestionId,
    pub score: Option<Score>,
    pub observations: Vec<QuestionResult>,
    pub has_conflict: bool,
}

/// The total grade written on the paper (e.g. the `Not:` cell), used only as a
/// cross-check against the sum of the question scores.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredTotal {
    pub value: ScoreValue,
    pub confidence: Confidence,
    pub source_region: SourceRegion,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExamPage {
    pub metadata: PageMetadata,
    pub question_results: Vec<QuestionResult>,
    pub declared_total: Option<DeclaredTotal>,
    pub model_version: Option<ModelVersion>,
    pub review_issues: Vec<ReviewIssue>,
}

impl ExamPage {
    pub fn new(metadata: PageMetadata, question_results: Vec<QuestionResult>) -> Self {
        Self {
            metadata,
            question_results,
            declared_total: None,
            model_version: None,
            review_issues: Vec::new(),
        }
    }

    pub fn with_declared_total(mut self, declared_total: DeclaredTotal) -> Self {
        self.declared_total = Some(declared_total);
        self
    }

    pub fn with_model_version(mut self, model_version: ModelVersion) -> Self {
        self.model_version = Some(model_version);
        self
    }

    pub fn with_review_issues(mut self, review_issues: Vec<ReviewIssue>) -> Self {
        self.review_issues = review_issues;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageFailure {
    pub page_index: u32,
    pub reason: String,
}

impl PageFailure {
    pub fn new(page_index: u32, reason: impl Into<String>) -> Self {
        Self {
            page_index,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningOutcomeResult {
    pub outcome_id: LearningOutcomeId,
    pub earned_score: ScoreValue,
    pub possible_score: ScoreValue,
    pub minimum_confidence: Option<Confidence>,
}

/// How question-to-outcome mappings without a printed weight are interpreted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutcomeAllocationRule {
    /// Every mapping needs an explicit weight; a missing weight stays unresolved.
    #[default]
    ExplicitWeightsOnly,
    /// For forms that print only `S1-PÇ2`: a question mapped to exactly one
    /// outcome, with no weight observed anywhere, carries the full weight.
    /// Questions with several unweighted outcomes still stay unresolved.
    SingleOutcomeImpliesFullWeight,
}

/// Cross-check between the handwritten total and the question scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotalCheck {
    NotDeclared,
    /// Not every exam question has an agreed score yet.
    Incomplete,
    Consistent,
    ConflictingDeclarations,
    Mismatch {
        declared: ScoreValue,
        computed: ScoreValue,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExamSession {
    pub id: String,
    pub exam: Option<Exam>,
    pub student: Option<StudentIdentity>,
    pub expected_page_count: Option<u32>,
    allocation_rule: OutcomeAllocationRule,
    pages: BTreeMap<u32, ExamPage>,
    failures: BTreeMap<u32, PageFailure>,
    pending_questions: BTreeMap<QuestionId, PendingQuestionDefinition>,
    pending_learning_outcomes: BTreeMap<LearningOutcomeId, PendingLearningOutcomeDefinition>,
    pending_outcome_mappings: BTreeMap<(QuestionId, LearningOutcomeId), PendingOutcomeMapping>,
    pending_student_identity: PendingStudentIdentity,
    canonical_question_results: BTreeMap<QuestionId, CanonicalQuestionResult>,
}

impl ExamSession {
    pub fn new(
        id: impl Into<String>,
        expected_page_count: Option<u32>,
    ) -> Result<Self, DomainError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(DomainError::EmptyValue("session id"));
        }
        Ok(Self {
            id,
            exam: None,
            student: None,
            expected_page_count,
            allocation_rule: OutcomeAllocationRule::default(),
            pages: BTreeMap::new(),
            failures: BTreeMap::new(),
            pending_questions: BTreeMap::new(),
            pending_learning_outcomes: BTreeMap::new(),
            pending_outcome_mappings: BTreeMap::new(),
            pending_student_identity: PendingStudentIdentity::default(),
            canonical_question_results: BTreeMap::new(),
        })
    }

    pub fn with_allocation_rule(mut self, rule: OutcomeAllocationRule) -> Self {
        self.allocation_rule = rule;
        self
    }

    pub fn allocation_rule(&self) -> OutcomeAllocationRule {
        self.allocation_rule
    }

    pub fn exam(&self) -> Option<&Exam> {
        self.exam.as_ref()
    }

    pub fn student(&self) -> Option<&StudentIdentity> {
        self.student.as_ref()
    }

    pub fn add_pending_question_definition(&mut self, discovery: PendingQuestionDefinition) {
        let pending = self
            .pending_questions
            .entry(discovery.question_id.clone())
            .or_insert_with(|| {
                PendingQuestionDefinition::new(discovery.question_id.clone(), None, None)
            });
        pending.add_candidate(discovery.resolved_maximum());
        for source_region in discovery.source_regions {
            pending.add_source_region(source_region);
        }
    }

    pub fn add_pending_learning_outcome(&mut self, discovery: PendingLearningOutcomeDefinition) {
        let pending = self
            .pending_learning_outcomes
            .entry(discovery.outcome_id.clone())
            .or_insert_with(|| {
                PendingLearningOutcomeDefinition::new(discovery.outcome_id.clone(), None, None)
            });
        for label in discovery.label_candidates {
            pending.add_candidate(Some(label));
        }
        pending.unreadable_label_observed |= discovery.unreadable_label_observed;
        for source_region in discovery.source_regions {
            pending.add_source_region(source_region);
        }
    }

    pub fn add_pending_outcome_mapping(&mut self, discovery: PendingOutcomeMapping) {
        let key = (discovery.question_id.clone(), discovery.outcome_id.clone());
        if self
            .exam
            .as_ref()
            .and_then(|exam| exam.questions.get(&discovery.question_id))
            .is_some_and(|question| {
                question.outcome_allocations.iter().any(|allocation| {
                    allocation.outcome_id == discovery.outcome_id
                        && discovery
                            .resolved_allocation()
                            .is_some_and(|weight| allocation.weight_basis_points == weight)
                })
            })
        {
            return;
        }
        let pending = self.pending_outcome_mappings.entry(key).or_insert_with(|| {
            PendingOutcomeMapping::new(
                discovery.question_id.clone(),
                discovery.outcome_id.clone(),
                None,
                None,
            )
        });
        for allocation in discovery.allocation_candidates {
            pending.add_candidate(Some(allocation));
        }
        pending.unreadable_allocation_observed |= discovery.unreadable_allocation_observed;
        for source_region in discovery.source_regions {
            pending.add_source_region(source_region);
        }
    }

    pub fn add_pending_student_name(
        &mut self,
        name: Option<String>,
        source_region: Option<SourceRegion>,
    ) {
        if name.as_deref().is_some_and(|candidate| {
            self.student
                .as_ref()
                .is_some_and(|student| candidate.trim() == student.name.trim())
        }) {
            return;
        }
        self.pending_student_identity.add_name(name, source_region);
    }

    pub fn add_pending_student_number(
        &mut self,
        number: Option<StudentNumber>,
        source_region: Option<SourceRegion>,
    ) {
        if number.as_ref().is_some_and(|candidate| {
            self.student
                .as_ref()
                .is_some_and(|student| candidate == &student.number)
        }) {
            return;
        }
        self.pending_student_identity
            .add_number(number, source_region);
    }

    pub fn pending_questions(&self) -> &BTreeMap<QuestionId, PendingQuestionDefinition> {
        &self.pending_questions
    }

    pub fn pending_learning_outcomes(
        &self,
    ) -> &BTreeMap<LearningOutcomeId, PendingLearningOutcomeDefinition> {
        &self.pending_learning_outcomes
    }

    pub fn pending_outcome_mappings(
        &self,
    ) -> &BTreeMap<(QuestionId, LearningOutcomeId), PendingOutcomeMapping> {
        &self.pending_outcome_mappings
    }

    pub fn pending_student_identity(&self) -> &PendingStudentIdentity {
        &self.pending_student_identity
    }

    pub fn canonical_question_results(&self) -> &BTreeMap<QuestionId, CanonicalQuestionResult> {
        &self.canonical_question_results
    }

    pub fn reconcile_pending_discoveries(&mut self) -> Result<(), DomainError> {
        if self.pending_student_identity.name_candidates.len() == 1
            && self.pending_student_identity.number_candidates.len() == 1
        {
            let name = self
                .pending_student_identity
                .name_candidates
                .first()
                .expect("one name candidate")
                .clone();
            let number = self
                .pending_student_identity
                .number_candidates
                .first()
                .expect("one number candidate")
                .clone();
            if let Ok(student) = StudentIdentity::new(name, number) {
                if self.resolve_student(student).is_ok() {
                    self.pending_student_identity = PendingStudentIdentity::default();
                }
            }
        }

        let existing = self.exam.clone();
        let mut outcomes: BTreeMap<LearningOutcomeId, LearningOutcome> = existing
            .as_ref()
            .map(|exam| exam.learning_outcomes.clone())
            .unwrap_or_default();
        for (id, pending) in &self.pending_learning_outcomes {
            let outcome = match pending.label_resolution() {
                LabelResolution::Labeled(label) => LearningOutcome::new(id.clone(), label)?,
                LabelResolution::Unlabeled => LearningOutcome::unlabeled(id.clone()),
                LabelResolution::Unresolved => continue,
            };
            let merged = match outcomes.get(id) {
                Some(existing) => match existing.merged_with(&outcome) {
                    Some(merged) => merged,
                    None => continue,
                },
                None => outcome,
            };
            outcomes.insert(id.clone(), merged);
        }

        let mut question_ids: BTreeSet<QuestionId> = self
            .pending_questions
            .iter()
            .filter_map(|(id, pending)| pending.resolved_maximum().map(|_| id.clone()))
            .collect();
        question_ids.extend(
            self.pending_outcome_mappings
                .keys()
                .filter(|(question_id, _)| {
                    existing
                        .as_ref()
                        .is_some_and(|exam| exam.questions.contains_key(question_id))
                        || self
                            .pending_questions
                            .get(question_id)
                            .is_some_and(|pending| pending.resolved_maximum().is_some())
                })
                .map(|(question_id, _)| question_id.clone()),
        );

        let mut questions = Vec::new();
        for question_id in question_ids {
            let existing_question = existing
                .as_ref()
                .and_then(|exam| exam.questions.get(&question_id));
            let pending_question = self.pending_questions.get(&question_id);
            let discovered_maximum =
                pending_question.and_then(|pending| pending.resolved_maximum());
            let maximum_score = match (existing_question, discovered_maximum) {
                (Some(question), Some(discovered)) if question.maximum_score != discovered => {
                    continue;
                }
                (Some(question), _) => question.maximum_score,
                (None, Some(discovered)) => discovered,
                (None, None) => continue,
            };

            let mappings: Vec<_> = self
                .pending_outcome_mappings
                .iter()
                .filter(|((mapped_question, _), _)| mapped_question == &question_id)
                .collect();
            let implies_full_weight = self.allocation_rule
                == OutcomeAllocationRule::SingleOutcomeImpliesFullWeight
                && mappings.len() == 1;
            let mut allocations = Vec::new();
            let mappings_resolved = !mappings.is_empty()
                && mappings.iter().all(|((_, outcome_id), mapping)| {
                    let weight = match mapping.resolved_allocation() {
                        Some(weight) => weight,
                        None if implies_full_weight && mapping.has_no_allocation_evidence() => {
                            FULL_WEIGHT_BASIS_POINTS
                        }
                        None => return false,
                    };
                    if !outcomes.contains_key(outcome_id) {
                        return false;
                    }
                    let Ok(allocation) = OutcomeAllocation::new(outcome_id.clone(), weight) else {
                        return false;
                    };
                    allocations.push(allocation);
                    true
                })
                && allocations
                    .iter()
                    .map(|allocation| u32::from(allocation.weight_basis_points))
                    .sum::<u32>()
                    == OutcomeAllocation::TOTAL_BASIS_POINTS;

            let allocations = if mappings_resolved {
                if let Some(question) = existing_question {
                    if !question.outcome_allocations.is_empty()
                        && question.outcome_allocations != allocations
                    {
                        continue;
                    }
                    if question.outcome_allocations == allocations {
                        continue;
                    }
                }
                allocations
            } else if existing_question.is_some() {
                continue;
            } else {
                Vec::new()
            };

            questions.push(Question::new(question_id, maximum_score, allocations)?);
        }

        let has_exam_discoveries = !outcomes.is_empty() || !questions.is_empty();
        if has_exam_discoveries {
            let (exam_id, title) = existing
                .as_ref()
                .map(|exam| (exam.id.clone(), exam.title.clone()))
                .unwrap_or_else(|| {
                    (
                        format!("exam-for-{}", self.id),
                        "Discovered exam".to_owned(),
                    )
                });
            let discovered = Exam::new(exam_id, title, questions, outcomes.into_values())?;
            self.merge_exam_discoveries(discovered)?;
        }

        if let Some(exam) = &self.exam {
            self.pending_questions.retain(|id, pending| {
                !pending.resolved_maximum().is_some_and(|maximum| {
                    exam.questions
                        .get(id)
                        .is_some_and(|question| question.maximum_score == maximum)
                })
            });
            self.pending_learning_outcomes.retain(|id, pending| {
                let Some(outcome) = exam.learning_outcomes.get(id) else {
                    return true;
                };
                match pending.label_resolution() {
                    LabelResolution::Labeled(label) => outcome.label.as_deref() != Some(label),
                    LabelResolution::Unlabeled => false,
                    LabelResolution::Unresolved => true,
                }
            });
            let implies_full_weight =
                self.allocation_rule == OutcomeAllocationRule::SingleOutcomeImpliesFullWeight;
            self.pending_outcome_mappings
                .retain(|(question_id, outcome_id), mapping| {
                    let Some(question) = exam.questions.get(question_id) else {
                        return true;
                    };
                    let confirmed = match mapping.resolved_allocation() {
                        Some(weight) => question.outcome_allocations.iter().any(|allocation| {
                            allocation.outcome_id == *outcome_id
                                && allocation.weight_basis_points == weight
                        }),
                        None => {
                            implies_full_weight
                                && mapping.has_no_allocation_evidence()
                                && matches!(
                                    question.outcome_allocations.as_slice(),
                                    [only] if only.outcome_id == *outcome_id
                                        && only.weight_basis_points == FULL_WEIGHT_BASIS_POINTS
                                )
                        }
                    };
                    !confirmed
                });
        }
        Ok(())
    }

    pub fn merge_exam_discoveries(&mut self, discovered: Exam) -> Result<(), DomainError> {
        let merged = match self.exam.as_ref() {
            Some(existing) => existing.merge_discoveries(discovered)?,
            None => discovered,
        };
        for page in self.pages.values() {
            validate_page_against_exam(page, &merged)?;
        }
        self.exam = Some(merged);
        Ok(())
    }

    pub fn resolve_student(&mut self, student: StudentIdentity) -> Result<(), DomainError> {
        match &self.student {
            Some(existing) if existing != &student => Err(DomainError::ConflictingStudentIdentity),
            Some(_) => Ok(()),
            None => {
                self.student = Some(student);
                Ok(())
            }
        }
    }

    pub fn checkpoint_page(&mut self, mut page: ExamPage) -> Result<(), DomainError> {
        let index = page.metadata.page_index;
        if self.pages.contains_key(&index) {
            return Err(DomainError::DuplicatePage(index));
        }
        validate_page_regions(&page)?;
        if let Some(exam) = &self.exam {
            validate_page_against_exam(&page, exam)?;
        }

        let mut canonical = self.canonical_question_results.clone();
        for observation in &page.question_results {
            match canonical.get_mut(&observation.question_id) {
                None => {
                    canonical.insert(
                        observation.question_id.clone(),
                        CanonicalQuestionResult {
                            question_id: observation.question_id.clone(),
                            score: Some(observation.score),
                            observations: vec![observation.clone()],
                            has_conflict: false,
                        },
                    );
                }
                Some(existing) => {
                    let previous_score = existing.score;
                    let conflict = existing.has_conflict
                        || previous_score.is_some_and(|score| score != observation.score);
                    existing.observations.push(observation.clone());
                    if conflict {
                        existing.score = None;
                        existing.has_conflict = true;
                        let candidate = previous_score
                            .map(|previous| {
                                format!(
                                    "{} / {}; observed {} / {}",
                                    previous.value(),
                                    previous.maximum(),
                                    observation.score.value(),
                                    observation.score.maximum()
                                )
                            })
                            .unwrap_or_else(|| {
                                format!(
                                    "additional observation {} / {}",
                                    observation.score.value(),
                                    observation.score.maximum()
                                )
                            });
                        page.review_issues.push(ReviewIssue::new(
                            ReviewReason::ConflictingQuestionScore,
                            format!("question {} score", observation.question_id),
                            Some(candidate),
                            Some(observation.source_region.clone()),
                        ));
                    }
                }
            }
        }
        if let (Some(declared), Some(computed)) = (
            page.declared_total.as_ref(),
            self.exam
                .as_ref()
                .and_then(|exam| computed_total(exam, &canonical)),
        ) {
            if declared.value != computed {
                page.review_issues.push(ReviewIssue::new(
                    ReviewReason::TotalMismatch,
                    "declared total",
                    Some(format!(
                        "declared {}; question scores sum to {computed}",
                        declared.value
                    )),
                    Some(declared.source_region.clone()),
                ));
            }
        }
        validate_page_regions(&page)?;
        self.canonical_question_results = canonical;
        self.pages.insert(index, page);
        self.failures.remove(&index);
        Ok(())
    }

    pub fn record_failure(&mut self, failure: PageFailure) {
        if !self.pages.contains_key(&failure.page_index) {
            self.failures.insert(failure.page_index, failure);
        }
    }

    pub fn pages(&self) -> &BTreeMap<u32, ExamPage> {
        &self.pages
    }

    pub fn failures(&self) -> &BTreeMap<u32, PageFailure> {
        &self.failures
    }

    pub fn missing_page_indexes(&self) -> BTreeSet<u32> {
        self.expected_page_count
            .map(|count| {
                (0..count)
                    .filter(|index| !self.pages.contains_key(index))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn aggregate_learning_outcomes(
        &self,
    ) -> Result<BTreeMap<LearningOutcomeId, LearningOutcomeResult>, DomainError> {
        let Some(exam) = &self.exam else {
            return Ok(BTreeMap::new());
        };
        let mut totals: BTreeMap<LearningOutcomeId, (ScoreValue, ScoreValue, Option<Confidence>)> =
            BTreeMap::new();
        for canonical in self.canonical_question_results.values() {
            let Some(score) = canonical.score else {
                continue;
            };
            let Some(question) = exam.questions.get(&canonical.question_id) else {
                continue;
            };
            let minimum_confidence = canonical
                .observations
                .iter()
                .map(|observation| observation.confidence.value())
                .min_by(f32::total_cmp)
                .map(|value| Confidence::new(value).expect("observation confidence is valid"));
            for allocation in &question.outcome_allocations {
                let entry = totals.entry(allocation.outcome_id.clone()).or_insert((
                    ScoreValue::ZERO,
                    ScoreValue::ZERO,
                    None,
                ));
                entry.0 = entry
                    .0
                    .checked_add(score.value().allocated(allocation.weight_basis_points)?)?;
                entry.1 = entry.1.checked_add(
                    question
                        .maximum_score
                        .allocated(allocation.weight_basis_points)?,
                )?;
                entry.2 = match (entry.2, minimum_confidence) {
                    (Some(previous), Some(current)) => Some(
                        Confidence::new(previous.value().min(current.value()))
                            .expect("minimum remains valid"),
                    ),
                    (Some(previous), None) => Some(previous),
                    (None, current) => current,
                };
            }
        }
        Ok(totals
            .into_iter()
            .map(
                |(outcome_id, (earned_score, possible_score, minimum_confidence))| {
                    (
                        outcome_id.clone(),
                        LearningOutcomeResult {
                            outcome_id,
                            earned_score,
                            possible_score,
                            minimum_confidence,
                        },
                    )
                },
            )
            .collect())
    }

    /// Compares every handwritten total with the sum of the agreed question
    /// scores. Totals are re-checked here because scores may arrive on later
    /// pages than the total itself.
    pub fn total_check(&self) -> TotalCheck {
        let declared: BTreeSet<ScoreValue> = self
            .pages
            .values()
            .filter_map(|page| page.declared_total.as_ref().map(|total| total.value))
            .collect();
        let declared = match declared.len() {
            0 => return TotalCheck::NotDeclared,
            1 => *declared.first().expect("one declared total"),
            _ => return TotalCheck::ConflictingDeclarations,
        };
        let Some(computed) = self
            .exam
            .as_ref()
            .and_then(|exam| computed_total(exam, &self.canonical_question_results))
        else {
            return TotalCheck::Incomplete;
        };
        if declared == computed {
            TotalCheck::Consistent
        } else {
            TotalCheck::Mismatch { declared, computed }
        }
    }

    pub fn review_issue_count(&self) -> usize {
        self.pages
            .values()
            .map(|page| page.review_issues.len())
            .sum()
    }

    pub fn pending_discovery_count(&self) -> usize {
        self.pending_questions.len()
            + self.pending_learning_outcomes.len()
            + self.pending_outcome_mappings.len()
            + usize::from(!self.pending_student_identity.is_empty())
    }

    pub fn needs_review(&self) -> bool {
        self.exam.is_none()
            || self.student.is_none()
            || !self.failures.is_empty()
            || self.pending_discovery_count() > 0
            || self
                .canonical_question_results
                .values()
                .any(|result| result.has_conflict)
            || self.exam.as_ref().is_some_and(|exam| {
                !exam.learning_outcomes.is_empty()
                    && exam
                        .questions
                        .values()
                        .any(|question| question.outcome_allocations.is_empty())
            })
            || self
                .pages
                .values()
                .any(|page| !page.review_issues.is_empty())
            || matches!(
                self.total_check(),
                TotalCheck::Mismatch { .. } | TotalCheck::ConflictingDeclarations
            )
    }
}

/// Sum of the agreed scores, available only once every exam question has one.
fn computed_total(
    exam: &Exam,
    canonical: &BTreeMap<QuestionId, CanonicalQuestionResult>,
) -> Option<ScoreValue> {
    if exam.questions.is_empty() {
        return None;
    }
    exam.questions.keys().try_fold(ScoreValue::ZERO, |sum, id| {
        let score = canonical.get(id)?.score?;
        sum.checked_add(score.value()).ok()
    })
}

fn validate_page_regions(page: &ExamPage) -> Result<(), DomainError> {
    let expected = page.metadata.page_index;
    for region in page
        .question_results
        .iter()
        .map(|result| &result.source_region)
        .chain(page.declared_total.iter().map(|total| &total.source_region))
        .chain(
            page.review_issues
                .iter()
                .filter_map(|issue| issue.source_region.as_ref()),
        )
    {
        if region.page_index != expected {
            return Err(DomainError::SourceRegionPageMismatch {
                expected,
                actual: region.page_index,
            });
        }
    }
    Ok(())
}

fn validate_page_against_exam(page: &ExamPage, exam: &Exam) -> Result<(), DomainError> {
    let mut seen_questions = BTreeSet::new();
    for result in &page.question_results {
        if !seen_questions.insert(result.question_id.clone()) {
            return Err(DomainError::DuplicateQuestion(result.question_id.clone()));
        }
        let question = exam
            .questions
            .get(&result.question_id)
            .ok_or(DomainError::UnknownQuestion(result.question_id.clone()))?;
        if result.score.maximum() != question.maximum_score {
            return Err(DomainError::QuestionMaximumMismatch {
                question: result.question_id.clone(),
                expected: question.maximum_score,
                actual: result.score.maximum(),
            });
        }
    }
    Ok(())
}
