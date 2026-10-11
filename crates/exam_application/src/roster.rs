use exam_core::domain::{DomainError, StudentNumber};
use std::collections::BTreeMap;

/// One enrolled student of the course the exam belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterEntry {
    pub number: StudentNumber,
    pub name: String,
}

/// How a recognised student number relates to the course roster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RosterMatch<'a> {
    Exact(&'a RosterEntry),
    /// Exactly one enrolled number is one edit (substitution, insertion or
    /// deletion) away: most likely a single misread digit. Still reviewed.
    NearMatch(&'a RosterEntry),
    /// Several enrolled numbers are equally close; none is chosen.
    Ambiguous(Vec<&'a RosterEntry>),
    NotFound,
}

/// The course enrollment list. Handwritten names are never recognised; the
/// student is identified by number and the name comes from here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StudentRoster {
    entries: BTreeMap<StudentNumber, RosterEntry>,
}

impl StudentRoster {
    pub fn new(entries: impl IntoIterator<Item = RosterEntry>) -> Result<Self, DomainError> {
        let mut by_number = BTreeMap::new();
        for entry in entries {
            if entry.name.trim().is_empty() {
                return Err(DomainError::EmptyValue("roster student name"));
            }
            if by_number.insert(entry.number.clone(), entry).is_some() {
                return Err(DomainError::InvalidValue("duplicate roster student number"));
            }
        }
        Ok(Self { entries: by_number })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn resolve(&self, recognised: &str) -> RosterMatch<'_> {
        let recognised = recognised.trim();
        if recognised.is_empty() {
            return RosterMatch::NotFound;
        }
        if let Some(entry) = StudentNumber::new(recognised)
            .ok()
            .and_then(|number| self.entries.get(&number))
        {
            return RosterMatch::Exact(entry);
        }
        let near: Vec<&RosterEntry> = self
            .entries
            .values()
            .filter(|entry| within_one_edit(recognised, entry.number.as_str()))
            .collect();
        match near.as_slice() {
            [] => RosterMatch::NotFound,
            [only] => RosterMatch::NearMatch(only),
            _ => RosterMatch::Ambiguous(near),
        }
    }
}

/// True when `a` and `b` differ by at most one substitution, insertion or deletion.
fn within_one_edit(a: &str, b: &str) -> bool {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let (shorter, longer) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    if longer.len() - shorter.len() > 1 {
        return false;
    }
    let prefix = shorter
        .iter()
        .zip(longer.iter())
        .take_while(|(x, y)| x == y)
        .count();
    if shorter.len() == longer.len() {
        shorter[prefix..]
            .iter()
            .skip(1)
            .eq(longer[prefix..].iter().skip(1))
    } else {
        shorter[prefix..].iter().eq(longer[prefix + 1..].iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster(numbers: &[&str]) -> StudentRoster {
        StudentRoster::new(numbers.iter().map(|number| RosterEntry {
            number: StudentNumber::new(*number).unwrap(),
            name: format!("Student {number}"),
        }))
        .unwrap()
    }

    #[test]
    fn edit_distance_one_covers_substitution_insertion_and_deletion() {
        assert!(within_one_edit("04451095728", "04451095728"));
        assert!(within_one_edit("04451095728", "04451095729"));
        assert!(within_one_edit("0445109728", "04451095728"));
        assert!(within_one_edit("044510957288", "04451095728"));
        assert!(!within_one_edit("04451095728", "04451095711"));
        assert!(!within_one_edit("044510928", "04451095728"));
    }

    #[test]
    fn exact_near_ambiguous_and_missing_numbers_are_distinguished() {
        let roster = roster(&["04237384350", "00724475861", "00724475811"]);
        assert!(
            matches!(roster.resolve(" 04237384350 "), RosterMatch::Exact(e) if e.number.as_str() == "04237384350")
        );
        assert!(
            matches!(roster.resolve("04237384950"), RosterMatch::NearMatch(e) if e.number.as_str() == "04237384350")
        );
        assert!(matches!(
            roster.resolve("0423738435"),
            RosterMatch::NearMatch(_)
        ));
        // One edit from both 00724475861 and 00724475811: no guess.
        assert!(matches!(roster.resolve("00724475821"), RosterMatch::Ambiguous(c) if c.len() == 2));
        assert_eq!(roster.resolve("99999999999"), RosterMatch::NotFound);
        assert_eq!(roster.resolve(""), RosterMatch::NotFound);
    }

    #[test]
    fn duplicate_numbers_and_blank_names_are_rejected() {
        let entry = |name: &str| RosterEntry {
            number: StudentNumber::new("1").unwrap(),
            name: name.to_owned(),
        };
        assert!(StudentRoster::new([entry("A"), entry("B")]).is_err());
        assert!(StudentRoster::new([entry(" ")]).is_err());
    }
}
