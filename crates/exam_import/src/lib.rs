//! Importing an exam definition from the instructor's Word file, following
//! the İnönü Mühendislik exam sheet conventions:
//!
//! - the header holds the term, exam type, course and the
//!   `Soru-PÇ İlişkisi` table: `(S1)-PÇ2 (S2)-PÇ1 ...` above the `Puan:` and
//!   `Not:` rows;
//! - each question text starts with `S.1.` / `S1` / `Soru 1` and states its
//!   points, e.g. `(20puan)` or `(3*10=30 puan)`.
//!
//! Nothing is guessed: a question whose points cannot be found is reported in
//! `warnings` and must be completed with `with_maximum_score` before
//! `to_exam`. Column widths of the score table are kept so the sheet reader
//! can place the handwritten score cells of any number of questions.

pub mod docx;

pub use docx::{read_docx, Document, ImportError};

use docx::{Part, Table};
use exam_core::domain::{
    Exam, LearningOutcome, LearningOutcomeId, OutcomeAllocation, Question, QuestionId, ScoreValue,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExamMetadata {
    /// e.g. `2025-2026`.
    pub academic_year: Option<String>,
    /// `Güz`, `Bahar` or `Yaz`.
    pub term: Option<String>,
    /// `Vize`, `Final`, `Bütünleme` or `Mazeret`.
    pub exam_kind: Option<String>,
    /// The `Dersin Kodu ve Adı` value as printed, e.g. `BİLM374-2022-YAPAY ZEKA`.
    pub course: Option<String>,
    pub course_code: Option<String>,
    pub course_name: Option<String>,
    pub department: Option<String>,
    pub instructor: Option<String>,
    pub date: Option<String>,
    /// Exam group letter, e.g. `A`.
    pub group: Option<String>,
}

/// Horizontal extent of a table column, in twips from the page's left edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnSpan {
    pub left_twips: u32,
    pub width_twips: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedQuestion {
    /// As printed in the table, e.g. `S1`.
    pub printed_id: String,
    pub id: QuestionId,
    pub outcomes: Vec<LearningOutcomeId>,
    pub maximum_score: Option<ScoreValue>,
    /// The text the maximum was read from, e.g. `(3*10=30 puan)`.
    pub maximum_source: Option<String>,
    /// The question's `Puan:` cell.
    pub score_column: ColumnSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportWarning {
    /// The table lists the question but no points were found in its text.
    MissingMaximumScore(QuestionId),
    /// Points were found only on sub-items and were added up.
    SummedSubQuestionPoints(QuestionId),
    /// Several PÇ without weights: the instructor must split the points.
    UnweightedOutcomes(QuestionId),
    /// A numbered question in the text that the table does not list.
    QuestionNotInTable(QuestionId),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedExam {
    pub metadata: ExamMetadata,
    pub questions: Vec<ImportedQuestion>,
    pub page_width_twips: u32,
    /// The handwritten `Not:` cell.
    pub total_column: ColumnSpan,
    pub warnings: Vec<ImportWarning>,
}

impl ImportedExam {
    pub fn from_docx(bytes: &[u8]) -> Result<Self, ImportError> {
        Self::from_document(&read_docx(bytes)?)
    }

    pub fn from_document(document: &Document) -> Result<Self, ImportError> {
        let parts: Vec<&Part> = document.headers.iter().chain([&document.body]).collect();
        let (table, label_row) = parts
            .iter()
            .flat_map(|part| &part.tables)
            .find_map(|table| {
                table
                    .rows
                    .iter()
                    .position(|row| {
                        row.first()
                            .is_some_and(|c| fold(&c.text()).starts_with("soru-p"))
                    })
                    .map(|index| (table, index))
            })
            .ok_or(ImportError::NoQuestionTable)?;
        let left =
            (document.left_margin_twips as i64 + i64::from(table.indent_twips)).max(0) as u32;

        let mut warnings = Vec::new();
        let texts = question_texts(&document.body);
        let mut questions = Vec::new();
        let mut x = left;
        let header = &table.rows[label_row];
        let widths = cell_widths(table, header);
        for (cell, width) in header.iter().zip(&widths) {
            let column = ColumnSpan {
                left_twips: x,
                width_twips: *width,
            };
            x += width;
            let Some((printed_id, outcomes)) = parse_mapping(&cell.text()) else {
                continue;
            };
            let id = QuestionId::new(&printed_id)
                .map_err(|e| ImportError::Incomplete(format!("{printed_id}: {e}")))?;
            let (maximum_score, maximum_source) = match texts.iter().find(|q| q.id == id) {
                Some(text) => {
                    if text.summed {
                        warnings.push(ImportWarning::SummedSubQuestionPoints(id.clone()));
                    }
                    (text.maximum, text.source.clone())
                }
                None => (None, None),
            };
            if maximum_score.is_none() {
                warnings.push(ImportWarning::MissingMaximumScore(id.clone()));
            }
            if outcomes.len() > 1 {
                warnings.push(ImportWarning::UnweightedOutcomes(id.clone()));
            }
            questions.push(ImportedQuestion {
                printed_id,
                id,
                outcomes,
                maximum_score,
                maximum_source,
                score_column: column,
            });
        }
        if questions.is_empty() {
            return Err(ImportError::NoQuestionTable);
        }
        for text in &texts {
            if !questions.iter().any(|q| q.id == text.id) {
                warnings.push(ImportWarning::QuestionNotInTable(text.id.clone()));
            }
        }

        // The `Not:` row: the cell after its label spans the question columns.
        let total_column = table
            .rows
            .iter()
            .find(|row| {
                row.first()
                    .is_some_and(|c| fold(&c.text()).starts_with("not"))
            })
            .and_then(|row| {
                let widths = cell_widths(table, row);
                let label = *widths.first()?;
                Some(ColumnSpan {
                    left_twips: left + label,
                    width_twips: widths.iter().skip(1).sum(),
                })
            })
            .unwrap_or(ColumnSpan {
                left_twips: questions[0].score_column.left_twips,
                width_twips: questions.iter().map(|q| q.score_column.width_twips).sum(),
            });

        Ok(Self {
            metadata: metadata(&parts),
            questions,
            page_width_twips: document.page_width_twips,
            total_column,
            warnings,
        })
    }

    /// Supplies a maximum the document does not state (and clears its warning).
    pub fn with_maximum_score(mut self, id: &QuestionId, maximum: ScoreValue) -> Self {
        if let Some(question) = self.questions.iter_mut().find(|q| &q.id == id) {
            question.maximum_score = Some(maximum);
            question.maximum_source = Some("entered by the instructor".to_owned());
            self.warnings
                .retain(|w| w != &ImportWarning::MissingMaximumScore(id.clone()));
        }
        self
    }

    /// The domain exam. A question with a single PÇ carries its full weight;
    /// one with several unweighted PÇ gets no allocation and stays in review.
    pub fn to_exam(&self, exam_id: &str, title: &str) -> Result<Exam, ImportError> {
        let mut outcomes: Vec<LearningOutcomeId> = Vec::new();
        let mut questions = Vec::new();
        for q in &self.questions {
            let maximum = q.maximum_score.ok_or_else(|| {
                ImportError::Incomplete(format!("no maximum score for {}", q.printed_id))
            })?;
            for outcome in &q.outcomes {
                if !outcomes.contains(outcome) {
                    outcomes.push(outcome.clone());
                }
            }
            let allocations = match q.outcomes.as_slice() {
                [only] => vec![OutcomeAllocation::new(only.clone(), 10_000)
                    .map_err(|e| ImportError::Incomplete(e.to_string()))?],
                _ => Vec::new(),
            };
            questions.push(
                Question::new(q.id.clone(), maximum, allocations)
                    .map_err(|e| ImportError::Incomplete(e.to_string()))?,
            );
        }
        Exam::new(
            exam_id,
            title,
            questions,
            outcomes.into_iter().map(LearningOutcome::unlabeled),
        )
        .map_err(|e| ImportError::Incomplete(e.to_string()))
    }
}

/// One character, case- and Turkish-diacritic-insensitive.
fn fold_char(c: char) -> char {
    match c {
        'İ' | 'I' | 'ı' | 'i' => 'i',
        'Ş' | 'ş' => 's',
        'Ğ' | 'ğ' => 'g',
        'Ü' | 'ü' => 'u',
        'Ö' | 'ö' => 'o',
        'Ç' | 'ç' => 'c',
        'Â' | 'â' => 'a',
        c => c.to_ascii_lowercase(),
    }
}

/// Label form for matching: folded and without whitespace.
fn fold(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_whitespace())
        .map(fold_char)
        .collect()
}

fn cell_widths(table: &Table, row: &[docx::Cell]) -> Vec<u32> {
    let mut grid = table.grid_twips.iter();
    row.iter()
        .map(|cell| {
            let spanned: u32 = grid.by_ref().take(cell.grid_span as usize).sum();
            if spanned > 0 {
                spanned
            } else {
                cell.width_twips.unwrap_or(0)
            }
        })
        .collect()
}

/// `(S1)- PÇ2` → (`S1`, [`PÇ2`]); `(S3)-PÇ1,PÇ4` → (`S3`, [`PÇ1`, `PÇ4`]).
fn parse_mapping(text: &str) -> Option<(String, Vec<LearningOutcomeId>)> {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let (question, outcomes) = compact.split_once('-')?;
    let question = question.trim_matches(|c| c == '(' || c == ')');
    let digits_at = question.find(|c: char| c.is_ascii_digit())?;
    if !question[..digits_at].eq_ignore_ascii_case("s")
        || !question[digits_at..].chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let outcomes: Vec<LearningOutcomeId> = outcomes
        .split([',', ';', '/', '+'])
        .filter(|o| !o.is_empty())
        .filter_map(|o| LearningOutcomeId::new(o).ok())
        .collect();
    (!outcomes.is_empty()).then(|| (question.to_owned(), outcomes))
}

struct QuestionText {
    id: QuestionId,
    maximum: Option<ScoreValue>,
    source: Option<String>,
    summed: bool,
}

/// `S.1.`, `S1`, `S.2`, `Soru 3`, `SORU-4)` at the start of a paragraph.
fn question_number(paragraph: &str) -> Option<QuestionId> {
    // Fold characters but keep spaces: `S.2 Aşağıdaki` must not read as `S.2a...`.
    let folded: String = paragraph
        .trim_start()
        .chars()
        .take(12)
        .map(fold_char)
        .collect();
    let rest = folded
        .strip_prefix("soru")
        .or_else(|| folded.strip_prefix('s'))?;
    let rest = rest.trim_start_matches(['.', '-', ':', ' ']);
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    let after = rest[digits.len()..].chars().next();
    if after.is_some_and(|c| c.is_alphanumeric()) {
        return None;
    }
    QuestionId::new(digits).ok()
}

fn question_texts(body: &Part) -> Vec<QuestionText> {
    let mut texts: Vec<QuestionText> = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let flush = |texts: &mut Vec<QuestionText>, block: &[&str]| {
        let Some(id) = block.first().and_then(|p| question_number(p)) else {
            return;
        };
        if texts.iter().any(|t| t.id == id) {
            return;
        }
        // Points stated with the question itself win; otherwise sub-items add up.
        let (maximum, source, summed) = match points(block[0]) {
            Some((value, source)) => (Some(value), Some(source), false),
            None => {
                let found: Vec<(ScoreValue, String)> =
                    block[1..].iter().filter_map(|p| points(p)).collect();
                match found.as_slice() {
                    [] => (None, None, false),
                    [(value, source)] => (Some(*value), Some(source.clone()), false),
                    many => (
                        many.iter()
                            .try_fold(ScoreValue::ZERO, |sum, (v, _)| sum.checked_add(*v).ok()),
                        Some(
                            many.iter()
                                .map(|(_, s)| s.as_str())
                                .collect::<Vec<_>>()
                                .join(" + "),
                        ),
                        true,
                    ),
                }
            }
        };
        texts.push(QuestionText {
            id,
            maximum,
            source,
            summed,
        });
    };
    for paragraph in &body.paragraphs {
        if question_number(paragraph).is_some() {
            flush(&mut texts, &block);
            block.clear();
        }
        if !block.is_empty() || question_number(paragraph).is_some() {
            block.push(paragraph);
        }
    }
    flush(&mut texts, &block);
    texts
}

fn parse_number(text: &str) -> Option<ScoreValue> {
    ScoreValue::from_decimal(&text.replace(',', ".")).ok()
}

fn numbers(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() || ((c == ',' || c == '.') && !current.is_empty()) {
            current.push(c);
        } else if !current.is_empty() {
            found.push(current.trim_end_matches([',', '.']).to_owned());
            current.clear();
        }
    }
    if !current.is_empty() {
        found.push(current.trim_end_matches([',', '.']).to_owned());
    }
    found
}

/// Points stated in a text: the last `(... puan)` group, else `N puan`.
/// `(3*10=30 puan)` → 30, `(3x10 puan)` → 30, `(12,5 Puan)` → 12.5.
pub fn points(text: &str) -> Option<(ScoreValue, String)> {
    let mut groups = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match c {
            '(' => start = Some(i),
            ')' => {
                if let Some(s) = start.take() {
                    groups.push(&text[s..=i]);
                }
            }
            _ => {}
        }
    }
    if let Some(group) = groups.into_iter().rev().find(|g| fold(g).contains("puan")) {
        let inner = group.trim_matches(|c| c == '(' || c == ')');
        let value = if let Some((_, after)) = inner.rsplit_once('=') {
            numbers(after).first().and_then(|n| parse_number(n))
        } else {
            let values: Vec<ScoreValue> = numbers(inner)
                .iter()
                .filter_map(|n| parse_number(n))
                .collect();
            match values.as_slice() {
                [only] => Some(*only),
                [a, b] if inner.contains(['*', 'x', 'X', '×']) => product(*a, *b),
                _ => None,
            }
        };
        return value.map(|v| (v, group.to_owned()));
    }
    // `... 15 puan` without parentheses: the number right before the word.
    let chars: Vec<char> = text.chars().collect();
    let folded: String = chars.iter().map(|&c| fold_char(c)).collect();
    let at = folded.find("puan")?;
    let before: String = chars[..folded[..at].chars().count()].iter().collect();
    let number = numbers(&before).pop()?;
    parse_number(&number).map(|v| (v, format!("{number} puan")))
}

/// Exact product of two scores (e.g. `3*10`), if it has at most two decimals.
fn product(a: ScoreValue, b: ScoreValue) -> Option<ScoreValue> {
    let micro = u128::from(a.micro_points()) * u128::from(b.micro_points()) / 1_000_000;
    (micro % 10_000 == 0)
        .then(|| u64::try_from(micro / 10_000).ok())
        .flatten()
        .and_then(|hundredths| ScoreValue::from_hundredths(hundredths).ok())
}

fn metadata(parts: &[&Part]) -> ExamMetadata {
    let mut meta = ExamMetadata::default();
    // Key/value cells of the header tables.
    let value_after = |label: &str| -> Option<String> {
        parts
            .iter()
            .flat_map(|p| &p.tables)
            .flat_map(|t| &t.rows)
            .find_map(|row| {
                let index = row.iter().position(|c| fold(&c.text()) == fold(label))?;
                row[index + 1..]
                    .iter()
                    .map(docx::Cell::text)
                    .find(|t| !t.is_empty())
            })
    };
    meta.course = value_after("Dersin Kodu ve Adı");
    meta.department = value_after("Bölümü");
    meta.date = value_after("Tarih/Saat");
    meta.instructor = value_after("Dersin Sorumlusu");
    if let Some(course) = &meta.course {
        let segments: Vec<&str> = course.split('-').map(str::trim).collect();
        meta.course_code = segments.first().map(|s| (*s).to_owned());
        if segments.len() > 1 {
            meta.course_name = segments.last().map(|s| (*s).to_owned());
        }
    }
    let paragraphs: Vec<&String> = parts.iter().flat_map(|p| &p.paragraphs).collect();
    if let Some(title) = paragraphs
        .iter()
        .find(|p| fold(p).contains("egitim-ogretimyili"))
    {
        let years: Vec<String> = numbers(title)
            .into_iter()
            .filter(|n| n.len() == 4)
            .collect();
        if years.len() >= 2 {
            meta.academic_year = Some(format!("{}-{}", years[0], years[1]));
        }
        let folded = fold(title);
        meta.term = [("guz", "Güz"), ("bahar", "Bahar"), ("yaz", "Yaz")]
            .iter()
            .find(|(key, _)| folded.contains(&format!("yili{key}")))
            .map(|(_, term)| (*term).to_owned());
        meta.exam_kind = [
            ("butunleme", "Bütünleme"),
            ("mazeret", "Mazeret"),
            ("final", "Final"),
            ("vize", "Vize"),
            ("arasinav", "Vize"),
        ]
        .iter()
        .find(|(key, _)| folded.contains(key))
        .map(|(_, kind)| (*kind).to_owned());
    }
    meta.group = paragraphs.iter().find_map(|p| {
        let folded = fold(p);
        let at = folded.find("grup")?;
        folded[at + 4..]
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_uppercase().to_string())
    });
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(text: &str) -> Option<String> {
        points(text).map(|(v, _)| v.to_string())
    }

    #[test]
    fn points_follow_the_written_conventions() {
        assert_eq!(
            score("Bir yapay zeka modeli ... test edilmiştir. (20puan)").as_deref(),
            Some("20")
        );
        assert_eq!(
            score("Aşağıdaki test sorularını çözünüz. (3*10=30 puan)").as_deref(),
            Some("30")
        );
        assert_eq!(
            score("Kısa cevaplayınız (3x10 puan)").as_deref(),
            Some("30")
        );
        assert_eq!(score("Hesaplayınız (12,5 Puan)").as_deref(), Some("12.5"));
        assert_eq!(
            score("(Model A) ile karşılaştırınız. (15 PUAN)").as_deref(),
            Some("15")
        );
        assert_eq!(
            score("Bu soru 25 puan değerindedir.").as_deref(),
            Some("25")
        );
        assert_eq!(score("Confusion matrix tablosunu oluşturunuz."), None);
    }

    #[test]
    fn question_numbers_are_recognised_without_false_starts() {
        for (text, id) in [
            ("S.1. Bir kanser", "1"),
            ("S.2 Aşağıdaki", "2"),
            ("S3) Açıklayınız", "3"),
            ("Soru 4: ", "4"),
            ("SORU-5", "5"),
        ] {
            assert_eq!(
                question_number(text),
                Some(QuestionId::new(id).unwrap()),
                "{text}"
            );
        }
        for text in ["Sınav süresi", "S.", "Sigmoid", "S1a şıkkı"] {
            assert_eq!(question_number(text), None, "{text}");
        }
    }

    #[test]
    fn mapping_cells_give_question_and_outcomes() {
        let (id, outcomes) = parse_mapping("(S1)- PÇ2").unwrap();
        assert_eq!(id, "S1");
        assert_eq!(outcomes, vec![LearningOutcomeId::new("PÇ2").unwrap()]);
        let (_, many) = parse_mapping("(S3)-PÇ1, PÇ4").unwrap();
        assert_eq!(many.len(), 2);
        assert!(parse_mapping("Soru-PÇ İlişkisi").is_none());
    }

    #[test]
    fn turkish_labels_fold_to_ascii() {
        assert_eq!(fold("Dersin Kodu ve Adı"), "dersinkoduveadi");
        assert_eq!(fold("EĞİTİM-ÖĞRETİM YILI"), "egitim-ogretimyili");
    }
}
