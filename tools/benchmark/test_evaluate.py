from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from evaluate import (
    EVALUATOR_VERSION,
    BenchmarkInputError,
    _normalize_question_id,
    evaluate,
    main,
    validate_document,
)


FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class BenchmarkEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.golden = validate_document(read_fixture("golden.synthetic.json"), "golden")
        self.predictions = validate_document(
            read_fixture("predictions.synthetic.json"), "predictions"
        )
        self.report = evaluate(self.golden, self.predictions)

    def test_synthetic_fixture_reports_separate_metrics_and_never_passes_gate(self) -> None:
        metrics = self.report["metrics"]
        self.assertEqual(self.report["gate_status"], "NOT_A_REAL_PAGE_BENCHMARK")
        self.assertEqual(self.report["counts"]["pages"], 3)
        self.assertEqual(metrics["schema_parse_rate"]["rate"], 1.0)
        self.assertEqual(metrics["student_number_exact_match"]["rate"], round(2 / 3, 6))
        self.assertEqual(metrics["question_id_exact_match"]["rate"], 1.0)
        self.assertEqual(metrics["score_exact_match"]["rate"], 0.4)
        self.assertEqual(metrics["maximum_score_exact_match"]["rate"], 1.0)
        self.assertEqual(metrics["question_to_outcome_mapping_accuracy"]["rate"], round(5 / 6, 6))
        self.assertEqual(metrics["review_recall"]["rate"], 1.0)
        self.assertEqual(metrics["review_precision"]["rate"], 1.0)
        self.assertEqual(metrics["bounding_box_coverage"]["rate"], round(2 / 3, 6))
        self.assertEqual(metrics["bounding_box_mean_iou"]["mean_iou"], 1.0)
        self.assertEqual(metrics["bounding_box_mean_iou"]["matched_valid_boxes"], 1)
        self.assertEqual(metrics["bounding_box_validity"]["rate"], 0.5)
        self.assertEqual(metrics["review_counts"], {"true_positive": 7, "false_positive": 0, "false_negative": 0})
        self.assertEqual(metrics["whole_page_perfect_extraction"]["rate"], 0.0)
        self.assertEqual(metrics["whole_session_perfect_extraction"]["rate"], 0.0)
        self.assertEqual(self.report["evaluator_version"], EVALUATOR_VERSION)

    def test_invalid_score_unknown_mapping_and_bad_region_are_reviewed(self) -> None:
        page = self.predictions["sessions"][0]["pages"][0]
        issue_fields = {issue["field_key"] for issue in page["review_issues"]}
        self.assertIn("question.earned_score", issue_fields)
        self.assertIn("outcome.id", issue_fields)
        self.assertIn("mapping", issue_fields)
        self.assertIn("source_region", issue_fields)
        self.assertEqual(page["questions"][0]["earned_score"], "18")
        self.assertEqual(page["questions"][0]["maximum_score"], "15")
        self.assertEqual(page["regions"][1]["page_index"], 4)
        self.assertEqual(self.report["metrics"]["review_false_negative_count"], 0)

    def test_conflicting_score_is_canonical_null_with_review_issue(self) -> None:
        page = self.predictions["sessions"][0]["pages"][1]
        question = next(item for item in page["questions"] if item["id"] == "Q6")
        self.assertIsNone(question["earned_score"])
        self.assertIn(
            {"field_key": "question.earned_score", "reason": "conflicting_question_score"},
            page["review_issues"],
        )

    def test_failed_parse_counts_as_failure_and_cannot_claim_field_accuracy(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        page = predictions["sessions"][1]["pages"][0]
        page.update(
            {
                "parse_status": "failed",
                "parse_error": "synthetic parser failure",
                "student": {"name": None, "number": None},
                "questions": [],
                "outcomes": [],
                "mappings": [],
                "regions": [],
                "confidence_by_field": {},
                "review_issues": [{"field_key": "page", "reason": "invalid_candidate"}],
            }
        )
        validated = validate_document(predictions, "predictions")
        report = evaluate(self.golden, validated)
        self.assertEqual(report["metrics"]["schema_parse_rate"]["rate"], round(2 / 3, 6))
        self.assertEqual(report["metrics"]["schema_parse_failures"], 1)
        self.assertEqual(report["metrics"]["score_exact_match"]["rate"], 0.4)
        self.assertEqual(report["metrics"]["review_counts"]["true_positive"], 5)
        self.assertEqual(report["metrics"]["review_false_negative_count"], 2)
        self.assertLess(report["metrics"]["review_recall"]["rate"], 1.0)

    def test_failed_parse_rejects_field_level_review_claims(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        page = predictions["sessions"][0]["pages"][1]
        page.update(
            {
                "parse_status": "failed",
                "parse_error": "synthetic parser failure",
                "student": {"name": None, "number": None},
                "questions": [],
                "outcomes": [],
                "mappings": [],
                "regions": [],
                "confidence_by_field": {},
                "review_issues": [
                    {"field_key": "question.earned_score", "reason": "invalid_candidate"}
                ],
            }
        )
        with self.assertRaisesRegex(BenchmarkInputError, "only request page-level review"):
            validate_document(predictions, "predictions")

    def test_gold_rejects_non_normalized_or_wrong_page_regions(self) -> None:
        golden = copy.deepcopy(self.golden)
        region = golden["sessions"][0]["pages"][0]["regions"][0]
        region["bbox"]["width"] = 2.0
        with self.assertRaisesRegex(BenchmarkInputError, "valid normalized boxes"):
            validate_document(golden, "golden")

    def test_golden_mapping_can_reference_entities_discovered_on_other_pages(self) -> None:
        document = validate_document(read_fixture("golden.synthetic.json"), "golden")
        pages = document["sessions"][0]["pages"]
        self.assertIn("Q3-b", {question["id"] for question in pages[0]["questions"]})
        self.assertNotIn("Q3-b", {question["id"] for question in pages[1]["questions"]})
        self.assertIn("LO-A", {outcome["id"] for outcome in pages[0]["outcomes"]})
        self.assertNotIn("LO-A", {outcome["id"] for outcome in pages[1]["outcomes"]})
        self.assertIn(
            {"question_id": "Q3-b", "outcome_id": "LO-A", "weight_basis_points": 10000},
            pages[1]["mappings"],
        )
        self.assertIn(
            {"question_id": "Q4", "outcome_id": "LO-B", "weight_basis_points": 10000},
            pages[0]["mappings"],
        )

    def test_golden_mapping_can_precede_question_defined_on_later_page(self) -> None:
        golden = copy.deepcopy(self.golden)
        pages = golden["sessions"][0]["pages"]
        question = next(item for item in pages[0]["questions"] if item["id"] == "Q3-b")
        pages[0]["questions"].remove(question)
        pages[1]["questions"].append(question)
        mapping = next(
            item for item in pages[1]["mappings"] if item["question_id"] == "Q3-b"
        )
        pages[1]["mappings"].remove(mapping)
        pages[0]["mappings"].append(mapping)
        self.assertEqual(validate_document(golden, "golden")["schema_version"], 1)

    def test_golden_mapping_can_reference_outcome_defined_on_another_page(self) -> None:
        golden = copy.deepcopy(self.golden)
        pages = golden["sessions"][0]["pages"]
        outcome = next(item for item in pages[0]["outcomes"] if item["id"] == "LO-A")
        pages[0]["outcomes"].remove(outcome)
        pages[1]["outcomes"].append(outcome)
        mapping = next(
            item for item in pages[1]["mappings"] if item["question_id"] == "Q3-b"
        )
        pages[1]["mappings"].remove(mapping)
        pages[0]["mappings"].append(mapping)
        self.assertEqual(validate_document(golden, "golden")["schema_version"], 1)

    def test_golden_mapping_rejects_references_unknown_in_its_session(self) -> None:
        golden = copy.deepcopy(self.golden)
        golden["sessions"][0]["pages"][1]["mappings"][0]["outcome_id"] = "LO-MISSING"
        with self.assertRaisesRegex(BenchmarkInputError, "unknown question/outcome in the session"):
            validate_document(golden, "golden")

    def test_unsupported_schema_version_fails_instead_of_being_ignored(self) -> None:
        golden = copy.deepcopy(self.golden)
        golden["schema_version"] = 2
        with self.assertRaisesRegex(BenchmarkInputError, "expected integer 1"):
            validate_document(golden, "golden")

    def test_schema_version_rejects_boolean_and_float_aliases(self) -> None:
        for version in (True, 1.0):
            with self.subTest(version=version):
                golden = copy.deepcopy(self.golden)
                golden["schema_version"] = version
                with self.assertRaisesRegex(BenchmarkInputError, "expected integer 1"):
                    validate_document(golden, "golden")

    def test_malformed_json_fails_with_nonzero_cli_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed_path = Path(directory) / "malformed.json"
            malformed_path.write_text("{", encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "--golden",
                        str(malformed_path),
                        "--predictions",
                        str(FIXTURES / "predictions.synthetic.json"),
                    ]
                )
        self.assertEqual(status, 2)
        self.assertIn("benchmark input error", stderr.getvalue())

    def test_synthetic_cli_evaluates_both_fixture_pairs(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["--synthetic"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["run_type"], "synthetic_fixture_validation")
        self.assertEqual(
            set(result["runs"]), {"perfect", "deliberately_faulty"}
        )

    def test_manifest_requires_reproducibility_timestamp_and_evaluator_version(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        del predictions["manifest"]["timestamp"]
        with self.assertRaisesRegex(BenchmarkInputError, "timestamp"):
            validate_document(predictions, "predictions")

    def test_older_evaluator_version_remains_compatible_with_same_schema(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["manifest"]["evaluator_version"] = "0.9.0"
        report = evaluate(self.golden, validate_document(predictions, "predictions"))
        self.assertEqual(report["manifest"]["evaluator_version"], "0.9.0")
        self.assertEqual(report["evaluator_version"], EVALUATOR_VERSION)

    def test_duplicate_question_ids_are_rejected_as_invalid_schema(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        questions = predictions["sessions"][0]["pages"][0]["questions"]
        questions.append(copy.deepcopy(questions[0]))
        with self.assertRaisesRegex(BenchmarkInputError, "duplicate question id"):
            validate_document(predictions, "predictions")

    def test_decimal_scores_compare_exactly_without_binary_float(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        page = predictions["sessions"][0]["pages"][0]
        page["questions"][0]["earned_score"] = "14.000"
        with self.assertRaisesRegex(BenchmarkInputError, "at most two decimals"):
            validate_document(predictions, "predictions")
        page["questions"][0]["earned_score"] = "14"
        validated = validate_document(predictions, "predictions")
        report = evaluate(self.golden, validated)
        self.assertEqual(report["metrics"]["score_exact_match"]["correct"], 3)

    def test_question_ids_follow_domain_trim_and_ascii_case_fold_policy(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        page = predictions["sessions"][0]["pages"][0]
        page["questions"][0]["id"] = " Q3-B "
        page["mappings"][0]["question_id"] = "(3-b)"
        validated = validate_document(predictions, "predictions")
        report = evaluate(self.golden, validated)
        self.assertEqual(report["metrics"]["question_id_exact_match"]["rate"], 1.0)
        self.assertEqual(report["metrics"]["score_exact_match"]["total"], 5)
        self.assertEqual(
            report["normalization"]["question_id"],
            "trim + ASCII A-Z case-fold + strip wrapping brackets and S/Soru/Q prefix (domain policy)",
        )

    def test_question_id_normalization_matches_the_rust_domain(self) -> None:
        # Keep in sync with crates/exam_core/tests/domain.rs.
        for spelling in ["1", "  S1 ", "s1", "(S1)", "S.1.", "S1:", "Soru 1", "SORU 1", "[1]"]:
            self.assertEqual(_normalize_question_id(spelling), "1", spelling)
        self.assertEqual(_normalize_question_id("Q3-b"), "3-b")
        for kept in ["1-a", "1.1", "A", "S", "Sa", "Soru"]:
            self.assertEqual(_normalize_question_id(kept), kept.lower())
        predictions = copy.deepcopy(self.predictions)
        predictions["sessions"][0]["pages"][0]["questions"][0]["id"] = "()"
        with self.assertRaisesRegex(BenchmarkInputError, "non-empty question id"):
            validate_document(predictions, "predictions")

    def test_student_numbers_are_not_normalized(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["sessions"][0]["pages"][0]["student"]["number"] = " SYN-001 "
        with self.assertRaisesRegex(BenchmarkInputError, "whitespace is not canonical"):
            validate_document(predictions, "predictions")

    def test_incorrect_allocation_lowers_its_exact_match_metric(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["sessions"][0]["pages"][0]["mappings"][0]["weight_basis_points"] = 9000
        report = evaluate(self.golden, validate_document(predictions, "predictions"))
        self.assertLess(
            report["metrics"]["allocation_exact_match"]["rate"],
            self.report["metrics"]["allocation_exact_match"]["rate"],
        )

    def test_overlapping_bounding_boxes_report_nontrivial_iou(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["sessions"][0]["pages"][0]["regions"][0]["bbox"]["x"] = 0.15
        report = evaluate(self.golden, validate_document(predictions, "predictions"))
        iou = report["metrics"]["bounding_box_mean_iou"]["mean_iou"]
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)
        self.assertEqual(iou, 0.6)

    def test_missing_bounding_box_reduces_coverage_without_zero_iou(self) -> None:
        golden = validate_document(read_fixture("golden.perfect.synthetic.json"), "golden")
        predictions = read_fixture("predictions.perfect.synthetic.json")
        predictions["sessions"][0]["pages"][0]["regions"] = []
        report = evaluate(golden, validate_document(predictions, "predictions"))
        self.assertEqual(report["metrics"]["bounding_box_coverage"]["rate"], 0.0)
        self.assertIsNone(report["metrics"]["bounding_box_mean_iou"]["mean_iou"])
        self.assertEqual(report["metrics"]["bounding_box_mean_iou"]["matched_valid_boxes"], 0)

    def test_invalid_bounding_box_lowers_validity_and_is_excluded_from_iou(self) -> None:
        golden = validate_document(read_fixture("golden.perfect.synthetic.json"), "golden")
        predictions = read_fixture("predictions.perfect.synthetic.json")
        predictions["sessions"][0]["pages"][0]["regions"][0]["bbox"]["x"] = 0.9
        report = evaluate(golden, validate_document(predictions, "predictions"))
        self.assertEqual(report["metrics"]["bounding_box_validity"]["rate"], 0.0)
        self.assertEqual(report["metrics"]["bounding_box_invalid_prediction_count"], 1)
        self.assertIsNone(report["metrics"]["bounding_box_mean_iou"]["mean_iou"])
        self.assertEqual(report["metrics"]["bounding_box_mean_iou"]["matched_valid_boxes"], 0)

    def test_unexpected_prediction_page_and_session_are_counted(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        extra_session = copy.deepcopy(predictions["sessions"][0])
        extra_session["session_id"] = "SYNTH-UNEXPECTED"
        predictions["sessions"].append(extra_session)
        report = evaluate(self.golden, validate_document(predictions, "predictions"))
        self.assertEqual(report["counts"]["unexpected_prediction_sessions"], 1)
        self.assertEqual(report["counts"]["unexpected_prediction_pages"], 2)

    def test_extra_prediction_page_prevents_whole_session_perfect(self) -> None:
        golden = validate_document(read_fixture("golden.perfect.synthetic.json"), "golden")
        predictions = read_fixture("predictions.perfect.synthetic.json")
        extra_page = copy.deepcopy(predictions["sessions"][0]["pages"][0])
        extra_page["page_index"] = 1
        for region in extra_page["regions"]:
            region["page_index"] = 1
        predictions["sessions"][0]["pages"].append(extra_page)
        report = evaluate(golden, validate_document(predictions, "predictions"))
        self.assertEqual(report["counts"]["unexpected_prediction_pages"], 1)
        self.assertEqual(report["metrics"]["whole_session_perfect_extraction"]["rate"], 0.0)

    def test_review_counts_use_explicit_ground_truth(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["sessions"][0]["pages"][0]["review_issues"] = []
        report = evaluate(self.golden, validate_document(predictions, "predictions"))
        self.assertEqual(
            report["metrics"]["review_counts"],
            {"true_positive": 3, "false_positive": 0, "false_negative": 4},
        )

    def test_empty_expected_metric_uses_null_rate(self) -> None:
        golden = copy.deepcopy(self.golden)
        predictions = copy.deepcopy(self.predictions)
        golden["sessions"] = [{"session_id": "EMPTY", "pages": [copy.deepcopy(golden["sessions"][0]["pages"][0])]}]
        predictions["sessions"] = [{"session_id": "EMPTY", "pages": [copy.deepcopy(predictions["sessions"][0]["pages"][1])]}]
        gold_page = golden["sessions"][0]["pages"][0]
        pred_page = predictions["sessions"][0]["pages"][0]
        gold_page.update(
            {
                "student": {"name": None, "number": None},
                "questions": [],
                "outcomes": [],
                "mappings": [],
                "regions": [],
                "review_required_fields": [],
            }
        )
        pred_page.update(
            {
                "student": {"name": None, "number": None},
                "questions": [],
                "outcomes": [],
                "mappings": [],
                "regions": [],
                "confidence_by_field": {},
                "review_issues": [],
            }
        )
        report = evaluate(
            validate_document(golden, "golden"),
            validate_document(predictions, "predictions"),
        )
        self.assertIsNone(report["metrics"]["student_number_exact_match"]["rate"])
        self.assertIsNone(report["metrics"]["score_exact_match"]["rate"])

    def test_perfect_synthetic_fixture_scores_every_page_and_session(self) -> None:
        golden = validate_document(read_fixture("golden.perfect.synthetic.json"), "golden")
        predictions = validate_document(
            read_fixture("predictions.perfect.synthetic.json"), "predictions"
        )
        report = evaluate(golden, predictions)
        self.assertEqual(report["metrics"]["whole_page_perfect_extraction"]["rate"], 1.0)
        self.assertEqual(report["metrics"]["whole_session_perfect_extraction"]["rate"], 1.0)
        self.assertEqual(report["metrics"]["bounding_box_mean_iou"]["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
