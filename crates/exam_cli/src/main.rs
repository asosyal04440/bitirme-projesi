//! Command-line tools around the exam pipeline.
//!
//! `exam-cli evaluate` runs the full on-device pipeline (sheet reader + the
//! application use case) over synthetic sheets from `tools/synthetic`, and
//! writes benchmark predictions plus an alignment/recognition summary:
//!
//! ```text
//! exam-cli evaluate --template tools/synthetic/templates/inonu-final-a.json \
//!     --model models/digit-crnn-v4.onnx --max-scores S1=20,S2=30,S3=25,S4=25 \
//!     --roster-from-labels --out data/synthetic/e2e/report data/synthetic/e2e/shard-*
//! ```

use exam_application::instrumentation::NoopInstrumentation;
use exam_application::ports::analyzer::{
    AnalyzerError, DocumentPage, ExamDocumentAnalyzer, StructuredPageExtraction,
};
use exam_application::roster::{RosterEntry, StudentRoster};
use exam_application::use_cases::ProcessExamPage;
use exam_core::domain::{
    ExamSession, ModelVersion, OutcomeAllocationRule, QuestionId, ReviewIssue, ReviewReason,
    ScoreValue, StudentNumber,
};
use exam_import::ImportedExam;
use exam_infrastructure::persistence::InMemorySessionRepository;
use exam_infrastructure::recognition::CellRecognizer;
use exam_infrastructure::sheet::{source_region, SheetField, SheetTemplate, TemplateSheetAnalyzer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage:
  exam-cli import <exam.docx> [--max-scores S3=25,S4=25]
  exam-cli evaluate --template <json> --model <onnx> (--exam <docx> | --max-scores S1=20,...)
                    [--max-scores ...] [--roster-from-labels] --out <dir> <synthetic-dir>...";

struct Options {
    template: Option<PathBuf>,
    model: Option<PathBuf>,
    exam: Option<PathBuf>,
    /// Maximum scores for the questions; with `--exam`, only the missing ones.
    max_scores: BTreeMap<QuestionId, ScoreValue>,
    roster_from_labels: bool,
    out: Option<PathBuf>,
    inputs: Vec<PathBuf>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        template: None,
        model: None,
        exam: None,
        max_scores: BTreeMap::new(),
        roster_from_labels: false,
        out: None,
        inputs: Vec::new(),
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || iter.next().cloned().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--template" => options.template = Some(PathBuf::from(value()?)),
            "--model" => options.model = Some(PathBuf::from(value()?)),
            "--exam" => options.exam = Some(PathBuf::from(value()?)),
            "--out" => options.out = Some(PathBuf::from(value()?)),
            "--max-scores" => {
                for item in value()?.split(',') {
                    let (id, score) = item
                        .split_once('=')
                        .ok_or(format!("bad max score {item}"))?;
                    options.max_scores.insert(
                        QuestionId::new(id).map_err(|e| e.to_string())?,
                        ScoreValue::from_decimal(score).map_err(|e| e.to_string())?,
                    );
                }
            }
            "--roster-from-labels" => options.roster_from_labels = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => options.inputs.push(PathBuf::from(other)),
        }
    }
    Ok(options)
}

fn imported_exam(
    path: &PathBuf,
    extra: &BTreeMap<QuestionId, ScoreValue>,
) -> Result<ImportedExam, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let exam = ImportedExam::from_docx(&bytes).map_err(|e| e.to_string())?;
    Ok(extra.iter().fold(exam, |exam, (id, maximum)| {
        exam.with_maximum_score(id, *maximum)
    }))
}

fn import(options: Options) -> Result<(), String> {
    let [path] = options.inputs.as_slice() else {
        return Err("import takes exactly one .docx file".to_owned());
    };
    let exam = imported_exam(path, &options.max_scores)?;
    let m = &exam.metadata;
    let report = json!({
        "metadata": {
            "academic_year": m.academic_year, "term": m.term, "exam_kind": m.exam_kind,
            "course": m.course, "course_code": m.course_code, "course_name": m.course_name,
            "department": m.department, "instructor": m.instructor, "date": m.date, "group": m.group,
        },
        "questions": exam.questions.iter().map(|q| json!({
            "id": q.printed_id,
            "outcomes": q.outcomes.iter().map(|o| o.as_str()).collect::<Vec<_>>(),
            "maximum_score": q.maximum_score.map(|v| v.to_string()),
            "maximum_source": q.maximum_source,
            "score_cell_twips": [q.score_column.left_twips, q.score_column.width_twips],
        })).collect::<Vec<_>>(),
        "warnings": exam.warnings.iter().map(|w| format!("{w:?}")).collect::<Vec<_>>(),
        "ready": exam.to_exam("check", "check").is_ok(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serializable")
    );
    Ok(())
}

/// Lets the use case consume an extraction computed beforehand.
struct Precomputed(StructuredPageExtraction);

impl ExamDocumentAnalyzer for Precomputed {
    fn analyze(&self, _: &DocumentPage) -> Result<StructuredPageExtraction, AnalyzerError> {
        Ok(self.0.clone())
    }
}

fn reason_key(reason: &ReviewReason) -> &'static str {
    match reason {
        ReviewReason::LowConfidence => "low_confidence",
        ReviewReason::ScoreExceedsMaximum => "score_exceeds_maximum",
        ReviewReason::UnknownQuestion => "unknown_question",
        ReviewReason::UnresolvedStudentIdentity => "unresolved_student_identity",
        ReviewReason::AmbiguousOutcomeMapping => "ambiguous_outcome_mapping",
        ReviewReason::AmbiguousCorrectedScore => "ambiguous_corrected_score",
        ReviewReason::InconsistentPageIdentity => "inconsistent_page_identity",
        ReviewReason::SourceRegionMismatch => "source_region_mismatch",
        ReviewReason::ConflictingQuestionScore => "conflicting_question_score",
        ReviewReason::InvalidCandidate => "invalid_candidate",
        ReviewReason::TotalMismatch => "total_mismatch",
        ReviewReason::BlankField => "blank_field",
        ReviewReason::RosterNearMatch => "roster_near_match",
    }
}

/// Benchmark review category of an application review issue.
fn review_category(issue: &ReviewIssue) -> &'static str {
    let field = issue.field.as_str();
    if issue.reason == ReviewReason::TotalMismatch || field.contains("total") {
        "declared_total"
    } else if field.contains("question") || field.contains("score") {
        "question.earned_score"
    } else if field.contains("student number") {
        "student.number"
    } else if field.contains("student") {
        "student.name"
    } else {
        "page"
    }
}

fn expected_text(cell: &Value) -> String {
    let written = cell["written"].as_str().unwrap_or("").replace(',', ".");
    if cell["struck"].is_null() || written.is_empty() {
        written
    } else {
        format!("~{written}")
    }
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x1, y1) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area = |r: [f64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter)
}

fn bounds(points: impl Iterator<Item = (f64, f64)>) -> [f64; 4] {
    points.fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, (x, y)| {
        [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
    })
}

fn kind_of(field_key: &str) -> &'static str {
    match field_key {
        "student.name" => "name",
        "student.number" => "number",
        "declared_total" => "total",
        _ => "score",
    }
}

fn utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[derive(Default)]
struct Tally {
    count: usize,
    hits: usize,
    sum: f64,
}

impl Tally {
    fn add(&mut self, hit: bool, value: f64) {
        self.count += 1;
        self.hits += usize::from(hit);
        self.sum += value;
    }

    fn report(&self, value_name: &str) -> Value {
        let rate = |x: f64| (x * 10_000.0).round() / 10_000.0;
        json!({
            "count": self.count,
            "rate": (self.count > 0).then(|| rate(self.hits as f64 / self.count as f64)),
            value_name: (self.count > 0).then(|| rate(self.sum / self.count as f64)),
        })
    }
}

fn evaluate(options: Options) -> Result<(), String> {
    let template_path = options.template.as_ref().ok_or("--template is required")?;
    let model_path = options.model.as_ref().ok_or("--model is required")?;
    let out = options.out.as_ref().ok_or("--out is required")?;
    if options.inputs.is_empty() {
        return Err("no synthetic directories given".to_owned());
    }
    let mut records = Vec::new();
    let mut golden_sessions = Vec::new();
    for input in &options.inputs {
        let labels = std::fs::read_to_string(input.join("labels.jsonl"))
            .map_err(|e| format!("{}: {e}", input.display()))?;
        for line in labels.lines().filter(|l| !l.trim().is_empty()) {
            let record: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
            records.push((input.clone(), record));
        }
        if let Ok(text) = std::fs::read_to_string(input.join("golden.json")) {
            let golden: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            golden_sessions.extend(golden["sessions"].as_array().cloned().unwrap_or_default());
        }
    }

    let roster = if options.roster_from_labels {
        let entries: BTreeMap<String, String> = records
            .iter()
            .filter_map(|(_, r)| {
                Some((
                    r["student"]["number"].as_str()?.to_owned(),
                    r["student"]["name"].as_str()?.to_owned(),
                ))
            })
            .collect();
        Some(
            StudentRoster::new(entries.into_iter().map(|(number, name)| RosterEntry {
                number: StudentNumber::new(number).expect("generated numbers are valid"),
                name,
            }))
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    let model_bytes =
        std::fs::read(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?;
    let recognizer = CellRecognizer::from_bytes(&model_bytes).map_err(|e| e.to_string())?;
    let template = SheetTemplate::load(template_path).map_err(|e| e.to_string())?;
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_owned();
    let model_version = ModelVersion::new(model_name.clone(), "1").map_err(|e| e.to_string())?;
    let analyzer = match &options.exam {
        Some(path) => TemplateSheetAnalyzer::for_imported_exam(
            template,
            recognizer,
            &imported_exam(path, &options.max_scores)?,
            model_version,
        ),
        None => TemplateSheetAnalyzer::new(
            template,
            recognizer,
            options.max_scores.clone(),
            model_version,
        ),
    }
    .map_err(|e| e.to_string())?;

    let mut sessions = Vec::new();
    let mut diagnostics = Vec::new();
    let mut alignment: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut recognition: BTreeMap<&str, Tally> = BTreeMap::new();
    let (mut identity, mut pages_found, mut elapsed) = (Tally::default(), 0_usize, 0.0_f64);
    for (directory, record) in &records {
        let sample_id = record["sample_id"]
            .as_str()
            .ok_or("record without sample_id")?;
        let image_name = record["image"]
            .as_str()
            .ok_or("labels were generated with --no-images")?;
        let photo = image::open(directory.join(image_name))
            .map_err(|e| e.to_string())?
            .to_rgb8();

        let started = Instant::now();
        let sheet = analyzer.read_sheet(&photo).map_err(|e| e.to_string())?;
        let extraction = analyzer.extraction(0, &sheet);
        let mut repository = InMemorySessionRepository::default();
        repository.insert(
            ExamSession::new(sample_id, Some(1))
                .map_err(|e| e.to_string())?
                .with_allocation_rule(OutcomeAllocationRule::SingleOutcomeImpliesFullWeight),
        );
        let precomputed = Precomputed(extraction.clone());
        let instrumentation = NoopInstrumentation;
        let mut use_case = ProcessExamPage::new(&precomputed, &mut repository, &instrumentation);
        if let Some(roster) = &roster {
            use_case = use_case.with_roster(roster);
        }
        let outcome = use_case.execute(sample_id, DocumentPage::new(0, vec![1]));
        elapsed += started.elapsed().as_secs_f64();
        pages_found += usize::from(sheet.page_found);
        let session = repository.get(sample_id).expect("session exists");

        // Alignment and raw recognition against the generator's ground truth.
        let truth: BTreeMap<&str, &Value> = record["cells"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| Some((c["field_key"].as_str()?, c)))
            .collect();
        for field in &sheet.fields {
            let Some(cell) = truth.get(field.field_key.as_str()) else {
                continue;
            };
            let expected = bounds(
                cell["quad"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|p| (p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0))),
            );
            let overlap = iou(bounds(field.quad.iter().map(|p| (p.x, p.y))), expected);
            alignment
                .entry(kind_of(&field.field_key))
                .or_default()
                .add(overlap >= 0.5, overlap);
            if let Some(reading) = &field.reading {
                let ok = reading.text == expected_text(cell);
                recognition
                    .entry(kind_of(&field.field_key))
                    .or_default()
                    .add(ok, f64::from(reading.confidence));
            }
        }
        diagnostics.push(json!({
            "sample_id": sample_id,
            "page_found": sheet.page_found,
            "registration_score": sheet.registration_score,
            "page_quad": sheet.page_quad.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        }));
        let resolved_number = session.student().map(|s| s.number.as_str().to_owned());
        if let Some(true_number) = record["student"]["number"].as_str() {
            identity.add(resolved_number.as_deref() == Some(true_number), 0.0);
        }

        // Benchmark prediction page.
        let number_reading = extraction
            .student
            .as_ref()
            .and_then(|s| s.student_number.candidate.clone());
        let mut confidence = Map::new();
        let mut regions = Vec::new();
        for field in &sheet.fields {
            if let Some(region) = source_region(0, &field.quad, sheet.width, sheet.height) {
                let b = region.bounds;
                regions.push(json!({"field_key": field.field_key, "page_index": 0,
                    "bbox": {"x": b.x, "y": b.y, "width": b.width, "height": b.height}}));
            }
            if let Some(reading) = &field.reading {
                if field.field != SheetField::StudentName {
                    confidence.insert(field.field_key.clone(), json!(reading.confidence));
                }
            }
        }
        let questions: Vec<Value> = extraction
            .questions
            .iter()
            .map(|q| {
                let earned = q
                    .score
                    .as_ref()
                    .and_then(|s| s.earned.candidate)
                    .map(|v| v.to_string());
                json!({"id": q.id.raw_text, "earned_score": earned,
                       "maximum_score": q.maximum_score.candidate.map(|v| v.to_string())})
            })
            .collect();
        let outcomes: Vec<Value> = extraction
            .learning_outcomes
            .iter()
            .map(|o| json!({"id": o.id.raw_text, "label": Value::Null}))
            .collect();
        let mappings: Vec<Value> = extraction
            .outcome_mappings
            .iter()
            .map(|m| {
                let weight = session.exam().and_then(|exam| {
                    let question = exam.questions.get(m.question_id.candidate.as_ref()?)?;
                    question
                        .outcome_allocations
                        .iter()
                        .find(|a| Some(&a.outcome_id) == m.outcome_id.candidate.as_ref())
                        .map(|a| a.weight_basis_points)
                });
                json!({"question_id": m.question_id.raw_text, "outcome_id": m.outcome_id.raw_text,
                       "weight_basis_points": weight})
            })
            .collect();
        let issues: BTreeSet<(&str, &str)> = match &outcome {
            Ok(page) => page
                .review_issues
                .iter()
                .map(|i| (review_category(i), reason_key(&i.reason)))
                .collect(),
            Err(_) => BTreeSet::from([("page", "invalid_candidate")]),
        };
        let mut issues: Vec<Value> = issues
            .into_iter()
            .map(|(f, r)| json!({"field_key": f, "reason": r}))
            .collect();
        // Fields the domain never accepted (e.g. a missing name without a roster)
        // still need a human; record them so review recall is not overstated.
        if session.student().is_none() {
            issues.push(
                json!({"field_key": "student.name", "reason": "unresolved_student_identity"}),
            );
        }
        sessions.push(json!({"session_id": sample_id, "pages": [{
            "page_index": 0,
            "student": {"name": session.student().map(|s| s.name.clone()),
                        "number": resolved_number.or(number_reading)},
            "questions": questions,
            "outcomes": outcomes,
            "mappings": mappings,
            "regions": regions,
            "parse_status": "ok",
            "parse_error": Value::Null,
            "confidence_by_field": confidence,
            "review_issues": issues,
        }]}));
    }

    let predictions = json!({
        "schema_version": 1,
        "manifest": {
            "model_name": model_name,
            "model_revision": "1",
            "artifact_sha256": Sha256::digest(&model_bytes).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "artifact_bytes": model_bytes.len(),
            "runtime_name": "exam-cli (tract-onnx)",
            "runtime_version": env!("CARGO_PKG_VERSION"),
            "backend": "cpu",
            "thread_count": 1,
            "temperature": 0,
            "seed": "not_applicable",
            "prompt_version": "none",
            "timestamp": utc_timestamp(),
            "evaluator_version": "1",
        },
        "sessions": sessions,
    });
    let golden =
        json!({"schema_version": 1, "corpus_type": "synthetic", "sessions": golden_sessions});
    let summary = json!({
        "sheets": records.len(),
        "page_outline_found": pages_found as f64 / records.len().max(1) as f64,
        "mean_seconds_per_sheet": elapsed / records.len().max(1) as f64,
        "alignment_iou": alignment.iter().map(|(k, t)| (k.to_string(), t.report("mean_iou"))).collect::<Map<_, _>>(),
        "raw_recognition": recognition.iter().map(|(k, t)| (k.to_string(), t.report("mean_confidence"))).collect::<Map<_, _>>(),
        "student_identity_resolved_correctly": identity.report("unused")["rate"],
        "roster_size": roster.as_ref().map(StudentRoster::len),
    });
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let write = |name: &str, value: &Value| {
        let path: PathBuf = out.join(name);
        std::fs::write(
            &path,
            serde_json::to_string_pretty(value).expect("serializable") + "\n",
        )
        .map_err(|e| format!("{}: {e}", path.display()))
    };
    write("predictions.json", &predictions)?;
    write("golden.json", &golden)?;
    write("summary.json", &summary)?;
    write("sheets.json", &Value::Array(diagnostics))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("serializable")
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("evaluate") => parse(&args[1..]).and_then(evaluate),
        Some("import") => parse(&args[1..]).and_then(import),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}
