# OCR benchmark contract (version 1)

This directory contains a model-independent evaluator. It does not load a model, read exam images, or call LiteRT/Paddle. The contract is the normalized boundary between any OCR runner and the current `StructuredPageExtraction` concepts.

Only synthetic or anonymized exam data belongs in benchmark inputs. Do not put real student names, numbers, grades, or page images in Git, and do not send identifiable pages to the commander chat. Model binaries stay in the local model cache; the run manifest records metadata and hashes only.

## Run

From the workspace root:

```powershell
python tools/benchmark/evaluate.py `
  --golden path/to/golden.json `
  --predictions path/to/predictions.json `
  --output path/to/report.json
```

The evaluator uses only the Python standard library. Invalid input exits with code 2 and a path-specific diagnostic. Without `--output`, the report is written to stdout. Reports never declare the feasibility gate passed: synthetic data is marked `NOT_A_REAL_PAGE_BENCHMARK`, and anonymized data is marked `METRICS_ONLY_NO_THRESHOLD_SET` until the commander and project owner establish acceptance thresholds.

Run both bundled synthetic evaluations (one perfect and one deliberately faulty) with:

```powershell
python -B tools/benchmark/evaluate.py --synthetic
```

Run the full deterministic test suite with:

```powershell
python -m unittest discover -s tools/benchmark -p "test_*.py" -v
```

The `fixtures/` directory has two synthetic pairs. The deliberately faulty pair covers 18/15, an unknown outcome, a missing student number, a low-confidence score, conflicting score observations already routed to review, and a malformed normalized source region. The perfect pair checks that a correct page/session can reach 100%. They are not exam samples and cannot establish model accuracy.

## File contract

Both documents use `schema_version: 1`; any unsupported version fails instead of being silently accepted. Documents group pages under stable, pseudonymous `session_id`s. Page indexes are zero-based. Each page contains:

- `student`: `name` and `number`, each a string or `null`;
- `questions`: unique `id`, decimal-string `earned_score`, and `maximum_score` (either score may be `null`);
- `outcomes`: unique `id` and optional `label`;
- `mappings`: `question_id`, `outcome_id`, and `weight_basis_points` (`null` means unresolved, never an equal split);
- `regions`: field key, page index, and normalized `{x,y,width,height}` box;
- the role-specific review and parse fields described below.

Scores are non-negative decimal strings with at most two fractional digits. They are compared as exact decimal values (`"18"` equals `"18.00"`); binary floating-point is not used for scores. Mapping weights are integer basis points from 1 through 10000. A `null` value means not extracted/unknown, not zero.

The golden document additionally requires `corpus_type` (`synthetic` or `anonymized_exam_pages`) and `review_required_fields`, an array of field categories that must reach human review. A mapping stays attached to the page where it was observed, but may reference a question or outcome discovered on another page in the same session; all golden references must resolve session-wide. Gold boxes must be valid normalized boxes on their source page.

The prediction document additionally requires a complete `manifest` with model name/revision, artifact SHA-256/size, runtime name/version, backend, thread count, temperature, seed (`not_applicable` when unsupported), prompt version, UTC timestamp, and evaluator version. The manifest version records the prediction producer; older evaluator versions remain valid when the schema version is supported. Each report separately identifies the current evaluator version. Every prediction page has `parse_status` (`ok` or `failed`), `parse_error` (`null` on success), `confidence_by_field` (values from 0 to 1), and `review_issues` (`field_key` plus a current `ReviewReason`-compatible reason). For a failed parse, all extracted values/regions/confidences must be empty and only page-level review (`field_key: page`) is permitted.

Question-ID comparison uses the current domain policy: Unicode whitespace trim, ASCII `A-Z` case-fold, removal of wrapping brackets and trailing `.`/`:`, and removal of an `S`/`Soru`/`Q`/`Question` prefix directly followed by a digit — so `(S1)`, `S.1.` and `Soru 1` all compare as `1`, while inner punctuation is preserved (`1-a`, `1.1` and `1a` stay distinct). This is reported in the output. Student numbers use exact string equality with no normalization. QIDs must be unique after domain normalization. Field keys used for confidence and regions use the normalized lower-ASCII form, e.g. `student.number`, `question:3-b.earned_score`, `outcome:LO-A.label`, and `mapping:3-b->LO-A.allocation`. Review keys are field categories such as `question.earned_score`, `mapping`, or `source_region`, matching the application review boundary rather than Paddle-specific tags.

## Metric definitions

- `student_number_exact_match` and `student_name_exact_match`: exact string equality over non-null gold values; a missing prediction is incorrect.
- `question_id_exact_match`: per-page equality of question-ID sets, including detection of extra IDs.
- `score_exact_match` and `maximum_score_exact_match`: exact decimal equality for each gold question with a known value, aligned by question ID.
- `outcome_id_exact_match`: per-page outcome-ID set equality; `outcome_label_exact_match` is exact for each known gold label.
- `question_to_outcome_mapping_accuracy`: intersection size divided by union size for question/outcome pairs, micro-aggregated over pages at their observed source location; referenced questions/outcomes may be discovered elsewhere in the session. Allocations have a separate exact-match metric.
- `field_coverage`: fraction of non-null gold scalar fields with a corresponding prediction; it measures presence, not correctness. `confidence_coverage` measures corresponding confidence values.
- `schema_parse_rate` and `schema_parse_failures`: prediction pages with `parse_status: ok` divided by expected pages. A missing prediction page is a parse failure.
- `review_precision`/`review_recall`: field-category precision/recall against the explicit golden `review_required_fields`. The report also includes TP/FP/FN counts; prediction mistakes do not silently rewrite review ground truth.
- `bounding_box_coverage`: expected field regions for which a prediction supplied a matching field key. `bounding_box_validity` checks finite normalized coordinates, positive area, bounds within `[0,1]`, and matching page index. `bounding_box_mean_iou` is computed only for matched, valid region pairs; missing/invalid regions remain visible in coverage/validity counts and are not converted into IoU zeroes.
- `whole_page_perfect_extraction`: exact student, question, score, outcome, mapping/allocation, and review-field agreement for a page. Regions are reported separately by IoU.
- `whole_session_perfect_extraction`: every expected page is perfect and the prediction has exactly the same page-index set for that session.

Rates with no eligible values are `null`, not zero. No aggregate score hides failures in a critical field. Do not choose pass thresholds from the synthetic fixture or the public table-demo smoke test.
